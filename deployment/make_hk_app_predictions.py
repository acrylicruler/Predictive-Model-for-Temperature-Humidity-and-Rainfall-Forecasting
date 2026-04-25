from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    from deployment.hk_best_model_config import (
        BASE_TARGETS,
        SUPERVISED_TARGETS,
        BEST_MODELS,
        HK_SPLIT,
        expected_cache_keys,
    )
except ImportError:
    from hk_best_model_config import (  # type: ignore
        BASE_TARGETS,
        SUPERVISED_TARGETS,
        BEST_MODELS,
        HK_SPLIT,
        expected_cache_keys,
    )

# ============================================================
# Paths
# ============================================================
OUT_DIR = Path("outputs")
DATA_DIR = Path("data")
CSV_DIR = DATA_DIR / "regional_features_csv"
MODEL_CACHE_DIR = OUT_DIR / "modelling_outputs" / "model_cache"
APP_DIR = OUT_DIR / "app"
APP_DIR.mkdir(parents=True, exist_ok=True)

OUT_PRED_PARQUET = APP_DIR / "hk_best_model_predictions.parquet"
OUT_PRED_CSV = APP_DIR / "hk_best_model_predictions.csv"
OUT_META_JSON = APP_DIR / "hk_best_model_metadata.json"

TRAIN_END = pd.Timestamp(HK_SPLIT["train_end"])
VAL_END = pd.Timestamp(HK_SPLIT["val_end"])
TEST_END = pd.Timestamp(HK_SPLIT["test_end"])

PREDICT_DATE_START = (VAL_END + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
PREDICT_DATE_END = TEST_END.strftime("%Y-%m-%d")

ID_COLS = {"date", "time", "city", "region_id", "target_date"}
DROP_ALWAYS = {"ssrd_sum", "month", "dayofyear"}

# ============================================================
# Notebook-aligned preprocessing helpers
# ============================================================
def ensure_datetime(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col])
    return d


def is_lag_roll(col: str) -> bool:
    return ("_lag" in col) or (("_mean" in col or "_sum" in col) and col.endswith("d"))


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    if "time" in d.columns:
        d["time"] = pd.to_datetime(d["time"])
    if "target_date" in d.columns:
        d["target_date"] = pd.to_datetime(d["target_date"])

    if "city" in d.columns:
        d["city"] = d["city"].astype(str).str.upper().astype("category")
    if "region_id" in d.columns:
        d["region_id"] = d["region_id"].astype(str).astype("category")

    for c in ["year", "month", "dayofyear"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype("Int16")

    return d


def preprocess_supervised(df: pd.DataFrame) -> pd.DataFrame:
    d = enforce_dtypes(df)
    if "year" not in d.columns:
        d["year"] = pd.to_datetime(d["time"]).dt.year.astype("int16")
    else:
        d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int16")
    return d


def make_next_day_supervised(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    d = ensure_datetime(df, time_col=time_col)
    d = d.sort_values(["city", "region_id", time_col]).copy()
    d["target_date"] = d[time_col] + pd.Timedelta(days=1)

    for t in BASE_TARGETS:
        d[f"{t}_tplus1"] = d.groupby(["city", "region_id"], observed=False)[t].shift(-1)

    d = d.dropna(subset=SUPERVISED_TARGETS).reset_index(drop=True)
    return d


def get_feature_cols(df: pd.DataFrame, level: int = 1, label_suffix: str = "_tplus1") -> list[str]:
    label_cols = [f"{t}{label_suffix}" for t in BASE_TARGETS]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feat = [c for c in num_cols if c not in ID_COLS and c not in label_cols and c not in DROP_ALWAYS]

    if level == 1:
        feat = [c for c in feat if not is_lag_roll(c)]
    elif level != 2:
        raise ValueError("level must be 1 or 2")

    return sorted(feat)


def drop_first_k_days_per_region(df: pd.DataFrame, k: int = 14, time_col: str = "time") -> pd.DataFrame:
    d = ensure_datetime(df, time_col=time_col)
    d = d.sort_values(["city", "region_id", time_col]).copy()
    d["_day_rank"] = d.groupby(["city", "region_id"], observed=False).cumcount()
    d = d[d["_day_rank"] >= k].drop(columns=["_day_rank"]).reset_index(drop=True)
    return d


def clean_level_valid_rows(
    df_supervised: pd.DataFrame,
    level: int = 1,
    time_col: str = "time",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    d = df_supervised.copy()
    if feature_cols is None:
        feature_cols = get_feature_cols(d, level=level)
    if level == 2:
        d = drop_first_k_days_per_region(d, k=14, time_col=time_col)
    d = d.dropna(subset=SUPERVISED_TARGETS + list(feature_cols)).reset_index(drop=True)
    return d


def split_by_city_and_timeblock(
    df: pd.DataFrame,
    train_cities: list[str],
    val_cities: list[str],
    test_cities: list[str],
    train_end: str,
    val_end: str,
    test_end: str,
    date_col: str = "target_date",
):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])

    train_end = pd.to_datetime(train_end)
    val_end = pd.to_datetime(val_end)
    test_end = pd.to_datetime(test_end)

    train_mask = d["city"].isin(train_cities) & (d[date_col] <= train_end)
    val_mask = d["city"].isin(val_cities) & (d[date_col] > train_end) & (d[date_col] <= val_end)
    test_mask = d["city"].isin(test_cities) & (d[date_col] > val_end) & (d[date_col] <= test_end)

    return d.loc[train_mask].copy(), d.loc[val_mask].copy(), d.loc[test_mask].copy()


def get_XY(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].astype(float).to_numpy()
    Y = df[SUPERVISED_TARGETS].astype(float).to_numpy()
    return X, Y

# ============================================================
# Cache resolution helpers
# ============================================================
def _safe_fname(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _old_cache_path(tag: str) -> Path:
    return MODEL_CACHE_DIR / f"{_safe_fname(tag)}.joblib"


def _new_cache_path(tag: str, keep: int = 80, hlen: int = 12) -> Path:
    safe = _safe_fname(tag)[:keep]
    h = hashlib.md5(tag.encode("utf-8")).hexdigest()[:hlen]
    return MODEL_CACHE_DIR / f"{safe}__{h}.joblib"


def resolve_cache_file(cache_key: str) -> Path:
    p_new = _new_cache_path(cache_key)
    if p_new.exists():
        return p_new

    p_old = _old_cache_path(cache_key)
    if p_old.exists():
        return p_old

    index_path = MODEL_CACHE_DIR / "cache_index.csv"
    if index_path.exists():
        idx = pd.read_csv(index_path)
        exact = idx[idx["tag"] == cache_key]
        if not exact.empty:
            candidate = MODEL_CACHE_DIR / str(exact.iloc[-1]["filename"])
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"Could not resolve model cache for key:\n{cache_key}\n"
        f"Checked: {p_new.name}, {p_old.name}, and cache_index.csv"
    )

# ============================================================
# Data loading
# ============================================================
def _read_regional_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "time" not in df.columns:
        raise ValueError(f"{path.name} missing 'time' column")
    if "region_id" not in df.columns:
        raise ValueError(f"{path.name} missing 'region_id' column")

    df["time"] = pd.to_datetime(df["time"])
    df["region_id"] = df["region_id"].astype(str)

    if "city" not in df.columns:
        if path.stem.upper() == "TEST_HK":
            df["city"] = "HK"
        else:
            raise ValueError(
                f"{path.name} has no 'city' column. Add it before deployment."
            )
    else:
        df["city"] = df["city"].astype(str).str.upper()

    return df


def load_hk_regional() -> pd.DataFrame:
    preferred = CSV_DIR / "test_HK.csv"
    if preferred.exists():
        df = _read_regional_csv(preferred)
        df = df[df["city"] == "HK"].copy()
        if df.empty:
            raise ValueError("test_HK.csv loaded but contains no HK rows")
        return df.reset_index(drop=True)

    csv_files = sorted(CSV_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in {CSV_DIR}")

    frames = [_read_regional_csv(p) for p in csv_files]
    full = pd.concat(frames, ignore_index=True)
    full = full[full["city"] == "HK"].copy()
    if full.empty:
        raise ValueError("No HK rows found in data/regional_features_csv")
    return full.reset_index(drop=True)


def build_hk_test_frame() -> pd.DataFrame:
    hk_raw = load_hk_regional()
    hk_supervised = make_next_day_supervised(hk_raw, time_col="time")
    hk_supervised = preprocess_supervised(hk_supervised)

    feat_l1 = get_feature_cols(hk_supervised, level=1)
    missing_l1 = [c for c in BEST_MODELS["t2m_mean"].feature_cols if c not in feat_l1]
    if missing_l1:
        raise ValueError(
            f"Notebook-aligned L1 features are still missing after preprocessing: {missing_l1}"
        )

    hk_clean = clean_level_valid_rows(
        hk_supervised,
        level=1,
        time_col="time",
        feature_cols=BEST_MODELS["t2m_mean"].feature_cols,
    )

    _, _, hk_test = split_by_city_and_timeblock(
        hk_clean,
        train_cities=["HK"],
        val_cities=["HK"],
        test_cities=["HK"],
        train_end=HK_SPLIT["train_end"],
        val_end=HK_SPLIT["val_end"],
        test_end=HK_SPLIT["test_end"],
        date_col="target_date",
    )

    hk_test = hk_test.reset_index(drop=True)
    return hk_test

# ============================================================
# Minimal custom class to restore TorchMLP from joblib
# ============================================================
class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_layer_sizes=(256, 128),
        activation="relu",
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        max_epochs=40,
        patience=5,
        dropout=0.1,
        scale_y=True,
        use_amp=True,
        random_state=42,
        verbose=0,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.dropout = dropout
        self.scale_y = scale_y
        self.use_amp = use_amp
        self.random_state = random_state
        self.verbose = verbose

    def _act(self):
        if self.activation == "relu":
            return nn.ReLU()
        if self.activation == "tanh":
            return nn.Tanh()
        raise ValueError(f"Unsupported activation: {self.activation}")

    def _build_net(self, input_dim: int, out_dim: int):
        layers = []
        in_dim = input_dim
        for h in self.hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(self._act())
            if self.dropout and self.dropout > 0:
                layers.append(nn.Dropout(float(self.dropout)))
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        raise NotImplementedError("Deployment class is for loading/predicting only.")

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)

        if not hasattr(self, "model_") or self.model_ is None:
            last_bias_key = sorted(self.state_dict_.keys())[-1]
            last_bias = self.state_dict_[last_bias_key]
            out_dim = int(last_bias.shape[0]) if last_bias.ndim == 1 else 1
            input_dim = X.shape[1]
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model_ = self._build_net(input_dim=input_dim, out_dim=out_dim).to(self.device_)
            self.model_.load_state_dict(self.state_dict_)

        self.model_.eval()
        ds = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(
            ds,
            batch_size=int(self.batch_size),
            shuffle=False,
            drop_last=False,
            pin_memory=(getattr(self, "device_", torch.device("cpu")).type == "cuda"),
        )

        preds = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device_, non_blocking=True)
                yp = self.model_(xb).detach().cpu().numpy()
                preds.append(yp)

        yhat = np.vstack(preds)
        if getattr(self, "y_scaler_", None) is not None:
            yhat = self.y_scaler_.inverse_transform(yhat)
        if yhat.shape[1] == 1:
            return yhat.ravel()
        return yhat

# ============================================================
# LSTM helpers aligned to notebook
# ============================================================
class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2, out_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        last = hn[-1]
        return self.head(last)


def scale_X_sequences(X, scaler: StandardScaler, fit: bool = False):
    n, t, f = X.shape
    X2 = X.reshape(-1, f)
    if fit:
        scaler.fit(X2)
    return scaler.transform(X2).reshape(n, t, f)


def scale_Y(Y, scaler: StandardScaler, fit: bool = False):
    if fit:
        scaler.fit(Y)
    return scaler.transform(Y)


@torch.no_grad()
def predict_model(model, loader, device):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        yp = model(xb)
        preds.append(yp.detach().cpu().numpy())
        trues.append(yb.detach().cpu().numpy())
    return np.vstack(preds), np.vstack(trues)


def rebuild_lstm_from_bundle(bundle, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = bundle["model_cfg"]
    model = LSTMRegressor(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        out_dim=cfg["out_dim"],
    ).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, device


def predict_with_lstm_bundle(bundle, X, Y, device=None):
    scaler_x = bundle["scaler_x"]
    scaler_y = bundle["scaler_y"]
    batch_size = int(bundle["train_cfg"]["batch_size"])

    Xs = scale_X_sequences(X, scaler_x, fit=False)
    Ys = scale_Y(Y, scaler_y, fit=False)

    loader = DataLoader(
        SeqDataset(Xs, Ys),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model, device = rebuild_lstm_from_bundle(bundle, device=device)
    pred_s, true_s = predict_model(model, loader, device)
    pred = scaler_y.inverse_transform(pred_s)
    true = scaler_y.inverse_transform(true_s)
    return pred, true


def build_sequences_for_lstm(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    time_col: str = "time",
    seq_len: int = 14,
):
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col])
    d = d.sort_values(["city", "region_id", time_col]).copy()

    X_list, Y_list, meta = [], [], []

    for (city, rid), g in d.groupby(["city", "region_id"], sort=False, observed=False):
        g = g.sort_values(time_col)
        Xg = g[feature_cols].to_numpy(dtype=float)
        Yg = g[target_cols].to_numpy(dtype=float)
        Tg = g[time_col].to_numpy()

        for i in range(seq_len - 1, len(g)):
            X_list.append(Xg[i - seq_len + 1 : i + 1, :])
            Y_list.append(Yg[i, :])
            meta.append({"city": city, "region_id": str(rid), "time": Tg[i]})

    X = np.stack(X_list) if X_list else np.zeros((0, seq_len, len(feature_cols)))
    Y = np.stack(Y_list) if Y_list else np.zeros((0, len(target_cols)))
    meta_df = pd.DataFrame(meta)
    if not meta_df.empty:
        meta_df["time"] = pd.to_datetime(meta_df["time"])
        meta_df["target_date"] = meta_df["time"] + pd.Timedelta(days=1)
    return X, Y, meta_df

# ============================================================
# Model loading
# ============================================================
def load_flat_model(cache_key: str):
    p = resolve_cache_file(cache_key)
    obj = joblib.load(p)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj.get("tune_info", {})
    return obj, {}


def load_lstm_bundle(cache_key: str):
    p = resolve_cache_file(cache_key)
    obj = joblib.load(p)
    if isinstance(obj, dict) and obj.get("kind") == "lstm_bundle":
        return obj["bundle"], obj.get("tune_info", {})
    if isinstance(obj, dict) and "bundle" in obj:
        return obj["bundle"], obj.get("tune_info", {})
    raise ValueError(f"Cache file for {cache_key} does not look like an LSTM bundle")

# ============================================================
# Prediction builders aligned to notebook logic
# ============================================================
def build_flat_prediction_table(df: pd.DataFrame, cache_key: str) -> pd.DataFrame:
    model, tune_info = load_flat_model(cache_key)
    feature_cols = BEST_MODELS["tp_sum_mm"].feature_cols
    missing = sorted(set(feature_cols + SUPERVISED_TARGETS) - set(df.columns))
    if missing:
        raise ValueError(f"Flat prediction table missing columns: {missing}")

    X, Y = get_XY(df, feature_cols)
    pred = np.asarray(model.predict(X))
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)

    out = df[["city", "region_id", "time", "target_date"]].copy().reset_index(drop=True)
    out["region_id"] = out["region_id"].astype(str)

    out["actual_t2m_mean_tplus1"] = Y[:, 0]
    out["actual_rh2m_mean_tplus1"] = Y[:, 1]
    out["actual_tp_sum_mm_tplus1"] = Y[:, 2]

    if pred.shape[1] >= 3:
        out["pred_tp_sum_mm_tplus1"] = pred[:, 2]
    else:
        out["pred_tp_sum_mm_tplus1"] = pred[:, 0]

    passthrough = [c for c in ["lat", "lon", "t2m_mean", "rh2m_mean", "tp_sum_mm"] if c in df.columns]
    for c in passthrough:
        out[c] = df[c].to_numpy()

    out["pred_source_tp"] = cache_key
    out["pred_family_tp"] = BEST_MODELS["tp_sum_mm"].family
    out["val_best_score_tp"] = tune_info.get("best_val_score", np.nan)
    return out


def build_lstm_prediction_table(df: pd.DataFrame, cache_key: str, target_base: str) -> pd.DataFrame:
    bundle, tune_info = load_lstm_bundle(cache_key)
    feature_cols = bundle["model_cfg"]["feature_cols"]
    seq_len = int(bundle["model_cfg"]["seq_len"])
    target_col = f"{target_base}_tplus1"

    X_seq, Y_seq, meta_df = build_sequences_for_lstm(
        df,
        feature_cols=feature_cols,
        target_cols=[target_col],
        time_col="time",
        seq_len=seq_len,
    )
    pred, true = predict_with_lstm_bundle(bundle, X_seq, Y_seq)

    out = meta_df.copy().reset_index(drop=True)
    out["region_id"] = out["region_id"].astype(str)
    out[f"pred_{target_base}_tplus1"] = pred[:, 0]
    out[f"actual_{target_base}_tplus1"] = true[:, 0]

    suffix = "t2m" if target_base == "t2m_mean" else "rh"
    out[f"pred_source_{suffix}"] = cache_key
    out[f"val_best_score_{suffix}"] = tune_info.get("best_val_score", np.nan)
    return out

# ============================================================
# Main
# ============================================================
def main():
    print("Loading Hong Kong regional data...")
    hk_test = build_hk_test_frame()
    print(f"HK test rows: {len(hk_test):,}")
    if len(hk_test) == 0:
        raise ValueError("HK test frame is empty after notebook-aligned preprocessing and split")

    cache_keys = expected_cache_keys()
    print("Resolved cache keys:")
    print(json.dumps(cache_keys, indent=2))

    tp_table = build_flat_prediction_table(hk_test, cache_keys["tp_sum_mm"])
    t2m_table = build_lstm_prediction_table(hk_test, cache_keys["t2m_mean"], "t2m_mean")
    rh_table = build_lstm_prediction_table(hk_test, cache_keys["rh2m_mean"], "rh2m_mean")

    final = tp_table.merge(
        t2m_table[
            [
                "city",
                "region_id",
                "time",
                "target_date",
                "pred_t2m_mean_tplus1",
                "pred_source_t2m",
                "val_best_score_t2m",
            ]
        ],
        on=["city", "region_id", "time", "target_date"],
        how="left",
    )
    final = final.merge(
        rh_table[
            [
                "city",
                "region_id",
                "time",
                "target_date",
                "pred_rh2m_mean_tplus1",
                "pred_source_rh",
                "val_best_score_rh",
            ]
        ],
        on=["city", "region_id", "time", "target_date"],
        how="left",
    )

    final["err_t2m_mean_tplus1"] = final["pred_t2m_mean_tplus1"] - final["actual_t2m_mean_tplus1"]
    final["err_rh2m_mean_tplus1"] = final["pred_rh2m_mean_tplus1"] - final["actual_rh2m_mean_tplus1"]
    final["err_tp_sum_mm_tplus1"] = final["pred_tp_sum_mm_tplus1"] - final["actual_tp_sum_mm_tplus1"]

    final = final.sort_values(["target_date", "region_id"]).reset_index(drop=True)

    final.to_parquet(OUT_PRED_PARQUET, index=False)
    final.to_csv(OUT_PRED_CSV, index=False)

    metadata = {
        "generated_from": "best HK-only models",
        "pipeline": "notebook-aligned next-day supervised + preprocess_supervised + clean_level_valid_rows + city_timeblock test split",
        "cache_keys": cache_keys,
        "targets": {
            k: {
                "family": v.family,
                "model_tag": v.model_tag,
                "hyperparams": v.hyperparams,
                "report_rmse": v.report_rmse,
                "report_r2": v.report_r2,
            }
            for k, v in BEST_MODELS.items()
        },
        "prediction_window": {
            "start": PREDICT_DATE_START,
            "end": PREDICT_DATE_END,
        },
        "output_parquet": str(OUT_PRED_PARQUET),
        "output_csv": str(OUT_PRED_CSV),
    }
    OUT_META_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved prediction parquet -> {OUT_PRED_PARQUET}")
    print(f"Saved prediction csv     -> {OUT_PRED_CSV}")
    print(f"Saved metadata json      -> {OUT_META_JSON}")


if __name__ == "__main__":
    main()
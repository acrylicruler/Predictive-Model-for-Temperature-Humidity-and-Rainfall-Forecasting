# 04_make_features_and_targets.py
import pandas as pd
from pathlib import Path

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

CITIES = ["HK","BKK","HCM","KL"]

BASE_TARGETS = ["t2m_mean","rh2m_mean","tp_sum_mm"]
EXTRA_FEATS = ["sp_mean","u10_mean","v10_mean","ssrd_sum"]

LEADS = [1, 7, 30]  # change later if needed

def safe_read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path, engine="fastparquet")
    except Exception:
        try:
            return pd.read_parquet(path, engine="pyarrow")
        except Exception as e:
            print(f"[WARN] Failed reading {path}: {e}")
            return None

def add_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["time"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["dayofyear"] = dt.dt.dayofyear
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    return df

def add_memory_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["city","region_id","time"]).copy()

    for col in BASE_TARGETS:
        for lag in [1, 3, 7, 14]:
            df[f"{col}_lag{lag}"] = df.groupby(["city","region_id"])[col].shift(lag)

        for win in [3, 7, 14]:
            df[f"{col}_mean{win}"] = (
                df.groupby(["city","region_id"])[col]
                  .rolling(win, min_periods=win)
                  .mean()
                  .reset_index(level=[0,1], drop=True)
            )

        if col == "tp_sum_mm":
            for win in [3, 7, 14]:
                df[f"{col}_sum{win}"] = (
                    df.groupby(["city","region_id"])[col]
                      .rolling(win, min_periods=win)
                      .sum()
                      .reset_index(level=[0,1], drop=True)
                )
    return df

def make_future_targets(df: pd.DataFrame, lead_days: int) -> pd.DataFrame:
    df = df.sort_values(["city","region_id","time"]).copy()
    for col in BASE_TARGETS:
        df[f"{col}_target_L{lead_days}"] = df.groupby(["city","region_id"])[col].shift(-lead_days)
    return df

def main():
    all_df = []

    for city in CITIES:
        path = OUT_DIR / f"regional_daily_{city}.parquet"
        df = safe_read_parquet(path)

        if df is None:
            print(f"[WARN] Missing/unreadable {path}. Skipping {city}.")
            continue
        if df.empty:
            print(f"[WARN] {city}: regional_daily is empty. Skipping.")
            continue

        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])
        if "city" not in df.columns:
            df["city"] = city

        all_df.append(df)

    if not all_df:
        print("[ERROR] No regional_daily files found. Run 03 first.")
        return

    data = pd.concat(all_df, ignore_index=True)

    # sanity
    required = {"time","city","region_id"} | set(BASE_TARGETS)
    missing = required - set(data.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    # keep core + extras if present
    keep = ["time","city","region_id"] + BASE_TARGETS + [c for c in EXTRA_FEATS if c in data.columns]
    data = data[keep].copy()

    data = add_time_feats(data)
    data = add_memory_feats(data)

    for L in LEADS:
        data = make_future_targets(data, L)

    # Save “full” (good for EDA even with NaNs at ends)
    out_full = OUT_DIR / "regional_features_targets_full.parquet"
    data.to_parquet(out_full, index=False, engine="fastparquet")
    print("Saved (full):", out_full, "rows:", len(data), "cols:", data.shape[1])

    # Save model-ready per lead
    base_required = ["time","city","region_id"] + BASE_TARGETS
    for L in LEADS:
        target_cols = [f"{c}_target_L{L}" for c in BASE_TARGETS]
        needed = base_required + target_cols
        model_df = data.dropna(subset=needed).copy()
        out = OUT_DIR / f"regional_features_targets_L{L}.parquet"
        model_df.to_parquet(out, index=False, engine="fastparquet")
        print(f"Saved (L{L} model-ready):", out, "rows:", len(model_df), "cols:", model_df.shape[1])

    print("\nDone. (Missing cities were skipped; partial cities still contribute.)")

if __name__ == "__main__":
    main()

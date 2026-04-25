from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any

BASE_TARGETS = ["t2m_mean", "rh2m_mean", "tp_sum_mm"]
SUPERVISED_TARGETS = [f"{t}_tplus1" for t in BASE_TARGETS]

# Exact L1 features used in your modelling pipeline
L1_FEATURES = [
    "doy_cos",
    "doy_sin",
    "rh2m_mean",
    "sp_mean",
    "ssrd_mj_m2",
    "t2m_mean",
    "tp_sum_mm",
    "u10_mean",
    "v10_mean",
    "year",
]

HK_SPLIT = {
    "split_mode": "city_timeblock",
    "train_cities": ["HK"],
    "val_cities": ["HK"],
    "test_city": "HK",
    "train_end": "2016-12-31",
    "val_end": "2020-12-31",
    "test_end": "2024-12-31",
}


@dataclass(frozen=True)
class BestModelSpec:
    target_base: str
    family: str
    source: str          # "lstm" or "flat"
    level: int
    mode: str            # "single" or "multi"
    feature_cols: list[str]
    model_tag: str
    target_index: int | None
    seq_len: int | None
    hyperparams: dict[str, Any]
    report_rmse: float
    report_r2: float


BEST_MODELS: dict[str, BestModelSpec] = {
    "t2m_mean": BestModelSpec(
        target_base="t2m_mean",
        family="LSTM",
        source="lstm",
        level=1,
        mode="single",
        feature_cols=L1_FEATURES,
        model_tag="L1-LSTMseq14:t2m_mean-single",
        target_index=0,
        seq_len=14,
        hyperparams={
            "hidden_dim": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "batch_size": 64,
            "epochs": 60,
            "patience": 5,
        },
        report_rmse=1.20353761,
        report_r2=0.943125367,
    ),
    "rh2m_mean": BestModelSpec(
        target_base="rh2m_mean",
        family="LSTM",
        source="lstm",
        level=1,
        mode="single",
        feature_cols=L1_FEATURES,
        model_tag="L1-LSTMseq14:rh2m_mean-single",
        target_index=1,
        seq_len=14,
        hyperparams={
            "hidden_dim": 128,
            "num_layers": 3,
            "dropout": 0.2,
            "lr": 0.001,
            "weight_decay": 1e-05,
            "batch_size": 128,
            "epochs": 40,
            "patience": 5,
        },
        report_rmse=5.625781535,
        report_r2=0.758519948,
    ),
    "tp_sum_mm": BestModelSpec(
        target_base="tp_sum_mm",
        family="MLP",
        source="flat",
        level=1,
        mode="multi",
        feature_cols=L1_FEATURES,
        model_tag="L1-TorchMLP-multi",
        target_index=2,
        seq_len=None,
        hyperparams={
            "model__hidden_layer_sizes": (128,),
            "model__activation": "relu",
            "model__lr": 0.001,
            "model__weight_decay": 0.0001,
            "model__batch_size": 512,
            "model__dropout": 0.2,
            "model__max_epochs": 60,
            "model__patience": 8,
        },
        report_rmse=128.2012865,
        report_r2=0.393723077,
    ),
}


def _safe_fname(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _city_list_str(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return "-".join([str(i) for i in x])
    return str(x)


def make_exp_slug(
    level: int,
    mode: str,
    split_mode: str,
    train_cities,
    val_cities,
    test_city,
    train_end: str,
    val_end: str,
    test_end: str,
    split_sig_hash: str,
    feat_sig: str,
    keep: int = 140,
) -> str:
    tr = _city_list_str(train_cities)
    va = _city_list_str(val_cities)
    te = _city_list_str(test_city)
    raw = (
        f"L{level}_{mode}_{split_mode}"
        f"__TR-{tr}__VA-{va}__TE-{te}"
        f"__trEnd{train_end}__vaEnd{val_end}__teEnd{test_end}"
        f"__split{split_sig_hash}__feat{feat_sig}"
    )
    safe = _safe_fname(raw)
    if len(safe) > keep:
        h = hashlib.md5(safe.encode("utf-8")).hexdigest()[:12]
        safe = safe[:keep] + f"__{h}"
    return safe


def feature_sig(feature_cols: list[str]) -> str:
    return hashlib.md5("|".join(feature_cols).encode("utf-8")).hexdigest()[:10]


def flat_split_sig_hash(level: int, mode: str, split_cfg: dict) -> str:
    split_sig = (
        f"{split_cfg['split_mode']}|"
        f"tr{split_cfg['train_end']}|"
        f"va{split_cfg['val_end']}|"
        f"te{split_cfg['test_end']}|"
        f"train{','.join(split_cfg['train_cities'])}|"
        f"val{','.join(split_cfg['val_cities'])}|"
        f"test{split_cfg['test_city']}|"
        f"L{level}|{mode}"
    )
    return hashlib.md5(split_sig.encode("utf-8")).hexdigest()[:10]


def lstm_split_sig_hash(level: int, mode: str, seq_len: int, split_cfg: dict) -> str:
    split_sig = (
        f"{split_cfg['split_mode']}|"
        f"tr{split_cfg['train_end']}|"
        f"va{split_cfg['val_end']}|"
        f"te{split_cfg['test_end']}|"
        f"train{','.join(split_cfg['train_cities'])}|"
        f"val{','.join(split_cfg['val_cities'])}|"
        f"test{split_cfg['test_city']}|"
        f"L{level}|lstm|{mode}|seq{seq_len}"
    )
    return hashlib.md5(split_sig.encode("utf-8")).hexdigest()[:10]


def build_cache_key(spec: BestModelSpec, split_cfg: dict = HK_SPLIT) -> str:
    feat_hash = feature_sig(spec.feature_cols)

    if spec.source == "flat":
        split_hash = flat_split_sig_hash(spec.level, spec.mode, split_cfg)
        exp_slug = make_exp_slug(
            level=spec.level,
            mode=spec.mode,
            split_mode=split_cfg["split_mode"],
            train_cities=split_cfg["train_cities"],
            val_cities=split_cfg["val_cities"],
            test_city=split_cfg["test_city"],
            train_end=split_cfg["train_end"],
            val_end=split_cfg["val_end"],
            test_end=split_cfg["test_end"],
            split_sig_hash=split_hash,
            feat_sig=feat_hash,
        )
    else:
        if spec.seq_len is None:
            raise ValueError("LSTM spec requires seq_len.")
        split_hash = lstm_split_sig_hash(spec.level, spec.mode, spec.seq_len, split_cfg)
        exp_slug = make_exp_slug(
            level=spec.level,
            mode=f"lstm_seq{spec.seq_len}_{spec.mode}",
            split_mode=split_cfg["split_mode"],
            train_cities=split_cfg["train_cities"],
            val_cities=split_cfg["val_cities"],
            test_city=split_cfg["test_city"],
            train_end=split_cfg["train_end"],
            val_end=split_cfg["val_end"],
            test_end=split_cfg["test_end"],
            split_sig_hash=split_hash,
            feat_sig=feat_hash,
        )

    return f"{spec.model_tag}__{exp_slug}"


def expected_cache_keys() -> dict[str, str]:
    return {target: build_cache_key(spec) for target, spec in BEST_MODELS.items()}


if __name__ == "__main__":
    from pprint import pprint
    pprint(expected_cache_keys())
# pipelines/sarima/utils.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple

import pandas as pd

from config.loader import load_config

def _features_root(cfg: dict, frequency: str = "daily") -> Path:
    # cfg['paths']['features'] jeśli masz; fallback do data/features
    features_base = Path(cfg.get("paths", {}).get("features", "data/features"))
    return features_base / "forex_daily" / f"frequency={frequency}"

def get_latest_ingest_date(features_root: Path) -> str:
    dates = []
    for p in features_root.glob("pair=*/ingest_date=*"):
        # .../pair=EUR_USD/ingest_date=2026-01-08
        ingest = p.name.split("=", 1)[1]
        dates.append(ingest)
    if not dates:
        raise RuntimeError(f"No ingest_date partitions found under: {features_root}")
    return max(dates)

def list_pairs_from_features(features_root: Path, ingest_date: str) -> List[str]:
    pairs = []
    for p in features_root.glob(f"pair=*/ingest_date={ingest_date}"):
        pair = p.parent.name.split("=", 1)[1]
        pairs.append(pair)
    return sorted(set(pairs))

def list_pairs(cfg: dict, features_root: Path, ingest_date: str) -> List[str]:
    # prefer config alpha_vantage.pairs if exists
    av_pairs = cfg.get("alpha_vantage", {}).get("pairs")
    if av_pairs:
        # [["EUR","USD"],["GBP","USD"],["USD","JPY"]] -> ["EUR_USD", ...]
        return [f"{a}_{b}" for a, b in av_pairs]
    return list_pairs_from_features(features_root, ingest_date)

def load_features_df(
    pair: str,
    ingest_date: Optional[str] = None,
    frequency: str = "daily",
) -> Tuple[pd.DataFrame, str]:
    cfg = load_config()
    root = _features_root(cfg, frequency=frequency)

    if ingest_date is None:
        ingest_date = get_latest_ingest_date(root)

    path = root / f"pair={pair}" / f"ingest_date={ingest_date}"
    if not path.exists():
        raise FileNotFoundError(f"Features path not found: {path}")

    df = pd.read_parquet(path)
    # date may be date type already, but ensure datetime for sorting
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # minimal sanity
    needed = {"date", "log_return", "target_return_1d"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in features for {pair}: {missing}")

    return df, ingest_date

def load_return_series(pair: str, ingest_date: Optional[str] = None, frequency: str = "daily") -> Tuple[pd.Series, str]:
    df, ingest_date = load_features_df(pair, ingest_date=ingest_date, frequency=frequency)
    s = pd.Series(df["log_return"].values, index=df["date"], name="log_return")
    return s, ingest_date

def load_xy(pair: str, ingest_date: Optional[str] = None, frequency: str = "daily") -> Tuple[pd.Series, pd.Series, str]:
    df, ingest_date = load_features_df(pair, ingest_date=ingest_date, frequency=frequency)
    y = pd.Series(df["log_return"].values, index=df["date"], name="log_return")
    y_tgt = pd.Series(df["target_return_1d"].values, index=df["date"], name="target_return_1d")
    return y, y_tgt, ingest_date

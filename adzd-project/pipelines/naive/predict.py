from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config.loader import load_config
from utils.logger import setup_logger
from pipelines.sarima.utils import list_pairs, _features_root, get_latest_ingest_date


def _load_features(cfg: dict, pair: str, ingest_date: str, frequency: str) -> pd.DataFrame:
    base = Path(cfg.get("paths", {}).get("features", "data/features")) / "forex_daily" / f"frequency={frequency}"
    p = base / f"pair={pair}" / f"ingest_date={ingest_date}"
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frequency", default="daily")
    ap.add_argument("--ingest-date", default=None)
    ap.add_argument("--pairs", default="all")
    ap.add_argument("--horizon", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()

    features_root = _features_root(cfg, frequency=args.frequency)
    ingest_date = args.ingest_date or get_latest_ingest_date(features_root)

    if args.pairs == "all":
        pairs = list_pairs(cfg, features_root, ingest_date)
    else:
        pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]

    horizon_cfg = cfg.get("models", {}).get("predict", {}).get(args.frequency, {}) or {}
    horizon = int(args.horizon or horizon_cfg.get("horizon_days", 1))

    logger = setup_logger("naive.predict", Path("logs/naive"))
    logger.info(f"NAIVE predict ingest_date={ingest_date}, pairs={pairs}, horizon={horizon}")

    run_ts = datetime.now(timezone.utc).isoformat()

    out_root = (
        Path(cfg.get("paths", {}).get("predictions", "data/predictions"))
        / "forex_daily"
        / f"frequency={args.frequency}"
        / "model=naive"
    )

    for pair in pairs:
        df = _load_features(cfg, pair, ingest_date, args.frequency)

        if "close" not in df.columns or "log_return" not in df.columns:
            logger.warning(f"Skipping {pair}: missing columns close/log_return in features")
            continue

        asof_ts = df["date"].iloc[-1]
        asof_date = asof_ts.date()
        last_close = float(df["close"].iloc[-1])

        # persistence na log_return: r_{t+1..t+h} = r_t
        last_ret = float(df["log_return"].dropna().iloc[-1])
        pred_rets = np.full(horizon, last_ret, dtype=float)

        pred_close_mean = last_close * np.exp(np.cumsum(pred_rets))
        pred_dates = pd.date_range(start=asof_ts + pd.Timedelta(days=1), periods=horizon, freq="D")

        out_df = pd.DataFrame(
            {
                "pair": pair,
                "ingest_date": ingest_date,
                "asof_date": asof_date,
                "step": np.arange(1, horizon + 1, dtype=int),
                "pred_date": pred_dates.date,
                "pred_return": pred_rets,
                "pred_close_mean": pred_close_mean,
                "pred_close_lower": np.nan,
                "pred_close_upper": np.nan,
                "model": "naive",
                "frequency": args.frequency,
                "run_ts_utc": run_ts,
            }
        )

        out_dir = out_root / f"pair={pair}" / f"ingest_date={ingest_date}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # czyścimy poprzedni zapis (żeby nie mieszać runów)
        for f in out_dir.glob("*.parquet"):
            f.unlink()

        out_df.to_parquet(out_dir / "predictions.parquet", index=False)

    logger.info("Predict done.")


if __name__ == "__main__":
    main()

# pipelines/prophet/backtest.py
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet

from config.loader import load_config
from utils.logger import setup_logger
from pipelines.sarima.utils import list_pairs, _features_root, get_latest_ingest_date, load_xy
from pipelines.common.metrics import mae, rmse, smape

LOGS_ROOT = Path("logs/prophet")


def _get_backtest_cfg(cfg: dict, frequency: str) -> tuple[int, int]:
    bt_root = cfg.get("models", {}).get("backtest", {}) or {}
    bt_cfg = bt_root.get(frequency) or bt_root.get("daily") or {}
    test_size = int(bt_cfg.get("test_size", 30))
    min_train_size = int(bt_cfg.get("min_train_size", 60))
    return test_size, min_train_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frequency", default="daily")
    ap.add_argument("--ingest-date", default=None)
    ap.add_argument("--pairs", default="all")
    ap.add_argument("--test-size", type=int, default=None, help="Override config test_size")
    ap.add_argument("--min-train-size", type=int, default=None, help="Override config min_train_size")
    args = ap.parse_args()

    cfg = load_config()
    root = _features_root(cfg, frequency=args.frequency)
    ingest_date = args.ingest_date or get_latest_ingest_date(root)

    if args.pairs == "all":
        pairs = list_pairs(cfg, root, ingest_date)
    else:
        pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]

    test_size, min_train_size = _get_backtest_cfg(cfg, args.frequency)
    if args.test_size is not None:
        test_size = int(args.test_size)
    if args.min_train_size is not None:
        min_train_size = int(args.min_train_size)

    pcfg = (cfg.get("models", {}).get("prophet", {}) or {}).get(args.frequency) or (cfg.get("models", {}).get("prophet", {}) or {}).get("daily") or {}

    run_ts = datetime.now(timezone.utc).isoformat()
    out_root = (
        Path(cfg.get("paths", {}).get("metrics", "data/metrics"))
        / "backtest"
        / f"frequency={args.frequency}"
        / "model=prophet"
    )

    logger = setup_logger("prophet.backtest", LOGS_ROOT)
    logger.info(
        f"Prophet backtest ingest_date={ingest_date}, pairs={pairs}, "
        f"test_size={test_size}, min_train_size={min_train_size}"
    )

    min_len = min_train_size + test_size + 1

    rows: list[dict] = []
    for pair in pairs:
        y, y_tgt, _ = load_xy(pair, ingest_date=ingest_date, frequency=args.frequency)

        if len(y) < min_len or len(y_tgt) < test_size:
            logger.warning(f"Skipping {pair}: too short series len(y)={len(y)} (need >= {min_len})")
            continue

        # Prophet uczy na y = log_return
        df_all = pd.DataFrame({"ds": pd.to_datetime(y.index), "y": y.values}).dropna()
        df_tgt = pd.DataFrame({"ds": pd.to_datetime(y_tgt.index), "y_true": y_tgt.values}).dropna()

        merged = df_all.merge(df_tgt, on="ds", how="inner").sort_values("ds")
        if len(merged) < min_len:
            logger.warning(f"Skipping {pair}: too few merged points len={len(merged)} (need >= {min_len})")
            continue

        train = merged.iloc[:-test_size].copy()
        test = merged.iloc[-test_size:].copy()

        m = Prophet(
            yearly_seasonality=bool(pcfg.get("yearly_seasonality", False)),
            weekly_seasonality=bool(pcfg.get("weekly_seasonality", True)),
            daily_seasonality=bool(pcfg.get("daily_seasonality", False)),
            seasonality_mode=str(pcfg.get("seasonality_mode", "additive")),
            changepoint_prior_scale=float(pcfg.get("changepoint_prior_scale", 0.05)),
        )

        m.fit(train[["ds", "y"]])

        pred = m.predict(test[["ds"]])
        y_pred = pred["yhat"].to_numpy(dtype=float)
        y_true = test["y_true"].to_numpy(dtype=float)

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            logger.warning(f"Skipping {pair}: no valid backtest points after mask")
            continue

        rows.append(
            {
                "pair": pair,
                "ingest_date": ingest_date,
                "n": int(len(y_true)),
                "mae": mae(y_true, y_pred),
                "rmse": rmse(y_true, y_pred),
                "smape": smape(y_true, y_pred),
                "model": "prophet",
                "frequency": args.frequency,
                "run_ts_utc": run_ts,
            }
        )

    if not rows:
        logger.warning("No backtest rows computed (all pairs skipped). Exiting with success.")
        return

    for r in rows:
        out_dir = out_root / f"pair={r['pair']}" / f"ingest_date={ingest_date}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(r, indent=2))
        pd.DataFrame([r]).to_parquet(out_dir / "metrics.parquet", index=False)

    logger.info(f"Computed metrics for pairs: {len(rows)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()

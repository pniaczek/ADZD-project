# pipelines/sarima/backtest.py
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config.loader import load_config
from utils.logger import setup_logger
from pipelines.sarima.utils import list_pairs, _features_root, get_latest_ingest_date, load_xy
from pipelines.common.metrics import mae, rmse, smape

LOGS_ROOT = Path("logs/sarima")


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

    mcfg = (cfg.get("models", {}).get("sarima", {}) or {}).get(args.frequency) or (cfg.get("models", {}).get("sarima", {}) or {}).get("daily") or {}
    order = tuple(mcfg.get("order", [1, 0, 1]))
    seasonal_order = tuple(mcfg.get("seasonal_order", [0, 0, 0, 0]))
    enforce_stationarity = bool(mcfg.get("enforce_stationarity", True))
    enforce_invertibility = bool(mcfg.get("enforce_invertibility", True))

    run_ts = datetime.now(timezone.utc).isoformat()
    out_root = (
        Path(cfg.get("paths", {}).get("metrics", "data/metrics"))
        / "backtest"
        / f"frequency={args.frequency}"
        / "model=sarima"
    )

    logger = setup_logger("sarima.backtest", LOGS_ROOT)
    logger.info(
        f"SARIMA backtest ingest_date={ingest_date}, pairs={pairs}, "
        f"test_size={test_size}, min_train_size={min_train_size}, "
        f"order={order}, seasonal_order={seasonal_order}"
    )

    min_len = min_train_size + test_size + 1

    rows: list[dict] = []
    for pair in pairs:
        y, y_tgt, _ = load_xy(pair, ingest_date=ingest_date, frequency=args.frequency)

        if len(y) < min_len or len(y_tgt) < test_size:
            logger.warning(f"Skipping {pair}: too short series len(y)={len(y)} (need >= {min_len})")
            continue

        # y = log_return, y_tgt = target_return_1d
        y_vals = y.values.astype(float)
        tgt_vals = y_tgt.values.astype(float)

        mask = np.isfinite(y_vals) & np.isfinite(tgt_vals)
        y_vals = y_vals[mask]
        tgt_vals = tgt_vals[mask]

        if len(y_vals) < min_len or len(tgt_vals) < test_size:
            logger.warning(f"Skipping {pair}: too few finite points after mask len={len(y_vals)} (need >= {min_len})")
            continue

        train = y_vals[:-test_size]
        test_true = tgt_vals[-test_size:]

        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )
        res = model.fit(disp=False)

        fc = res.get_forecast(steps=len(test_true))
        test_pred = np.asarray(fc.predicted_mean, dtype=float)

        m2 = np.isfinite(test_true) & np.isfinite(test_pred)
        test_true = test_true[m2]
        test_pred = test_pred[m2]

        if len(test_true) == 0:
            logger.warning(f"Skipping {pair}: no valid backtest points after final mask")
            continue

        rows.append(
            {
                "pair": pair,
                "ingest_date": ingest_date,
                "n": int(len(test_true)),
                "mae": mae(test_true, test_pred),
                "rmse": rmse(test_true, test_pred),
                "smape": smape(test_true, test_pred),
                "model": "sarima",
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

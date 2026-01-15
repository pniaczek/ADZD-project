# pipelines/naive/backtest.py
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.loader import load_config
from utils.logger import setup_logger
from pipelines.sarima.utils import list_pairs, _features_root, get_latest_ingest_date, load_xy
from pipelines.common.metrics import mae, rmse, smape


def _get_backtest_cfg(cfg: dict, frequency: str) -> tuple[int, int]:
    """
    Returns (test_size, min_train_size) from config.
    Preferred path: models.backtest.<frequency>
    Fallback: models.backtest.daily
    Fallback: defaults (30, 60)
    """
    bt_root = cfg.get("models", {}).get("backtest", {}) or {}
    bt_cfg = bt_root.get(frequency) or bt_root.get("daily") or {}
    test_size = int(bt_cfg.get("test_size", 30))
    min_train_size = int(bt_cfg.get("min_train_size", 60))
    return test_size, min_train_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frequency", default="daily")
    ap.add_argument("--ingest-date", default=None)
    ap.add_argument("--pairs", default="all", help="all or comma-separated e.g. EUR_USD,GBP_USD")
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

    run_ts = datetime.now(timezone.utc).isoformat()

    out_root = (
        Path(cfg.get("paths", {}).get("metrics", "data/metrics"))
        / "backtest"
        / f"frequency={args.frequency}"
        / "model=naive"
    )

    logs_root = Path("logs/naive")
    logger = setup_logger("naive.backtest", logs_root)

    logger.info(
        f"NAIVE backtest ingest_date={ingest_date}, pairs={pairs}, "
        f"test_size={test_size}, min_train_size={min_train_size}"
    )

    min_len = min_train_size + test_size + 1

    rows: list[dict] = []
    for pair in pairs:
        y, y_tgt, _ = load_xy(pair, ingest_date=ingest_date, frequency=args.frequency)

        # Naive baseline dla target_return_1d:
        # pred_t = y(t) i porównujemy do target(t) = y(t+1)
        if len(y) < min_len or len(y_tgt) < test_size:
            logger.warning(f"Skipping {pair}: too short series len(y)={len(y)} (need >= {min_len})")
            continue

        df_len = len(y_tgt)
        y_pred = y.iloc[:df_len].values.astype(float)
        y_true = y_tgt.values.astype(float)

        # backtest tylko na końcówce
        y_pred_bt = y_pred[-test_size:]
        y_true_bt = y_true[-test_size:]

        mask = np.isfinite(y_true_bt) & np.isfinite(y_pred_bt)
        y_true_bt = y_true_bt[mask]
        y_pred_bt = y_pred_bt[mask]

        if len(y_true_bt) == 0:
            logger.warning(f"Skipping {pair}: no finite points after mask")
            continue

        rows.append(
            {
                "pair": pair,
                "ingest_date": ingest_date,
                "n": int(len(y_true_bt)),
                "mae": mae(y_true_bt, y_pred_bt),
                "rmse": rmse(y_true_bt, y_pred_bt),
                "smape": smape(y_true_bt, y_pred_bt),
                "model": "naive",
                "frequency": args.frequency,
                "run_ts_utc": run_ts,
            }
        )

    if not rows:
        logger.warning("No backtest rows computed (all pairs skipped). Exiting with success.")
        return

    # zapis per pair
    for r in rows:
        out_dir = out_root / f"pair={r['pair']}" / f"ingest_date={ingest_date}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(r, indent=2))
        pd.DataFrame([r]).to_parquet(out_dir / "metrics.parquet", index=False)

    logger.info(f"Computed metrics for pairs: {len(rows)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()

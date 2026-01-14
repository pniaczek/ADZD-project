from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.loader import load_config
from utils.logger import setup_logger
from pipelines.sarima.utils import list_pairs, _features_root, get_latest_ingest_date


def _features_path(cfg: dict, frequency: str, pair: str, ingest_date: str) -> Path:
    return (
        Path(cfg.get("paths", {}).get("features", "data/features"))
        / "forex_daily"
        / f"frequency={frequency}"
        / f"pair={pair}"
        / f"ingest_date={ingest_date}"
    )


def _predictions_path(cfg: dict, frequency: str, model: str, pair: str, ingest_date: str) -> Path:
    return (
        Path(cfg.get("paths", {}).get("predictions", "data/predictions"))
        / "forex_daily"
        / f"frequency={frequency}"
        / f"model={model}"
        / f"pair={pair}"
        / f"ingest_date={ingest_date}"
        / "predictions.parquet"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frequency", default="daily")
    ap.add_argument("--ingest-date", default=None)
    ap.add_argument("--pairs", default="all")
    ap.add_argument("--models", default="naive,sarima,prophet")
    ap.add_argument("--history-days", type=int, default=120)

    # NOWE: kontrola rysowania interwałów
    ap.add_argument(
        "--no-intervals",
        action="store_true",
        help="Do not draw prediction intervals (only mean/close trajectory).",
    )

    args = ap.parse_args()

    cfg = load_config()

    features_root = _features_root(cfg, frequency=args.frequency)
    ingest_date = args.ingest_date or get_latest_ingest_date(features_root)

    if args.pairs == "all":
        pairs = list_pairs(cfg, features_root, ingest_date)
    else:
        pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    logger = setup_logger("visualize.forecast_vs_actual", Path("logs/visualize"))
    logger.info(
        f"Visualization ingest_date={ingest_date}, pairs={pairs}, models={models}, "
        f"intervals={'OFF' if args.no_intervals else 'ON'}"
    )

    out_root = (
        Path("reports/figures/forex_daily")
        / f"frequency={args.frequency}"
        / f"ingest_date={ingest_date}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for pair in pairs:
        fpath = _features_path(cfg, args.frequency, pair, ingest_date)
        if not fpath.exists():
            logger.warning(f"Skipping {pair}: features path missing: {fpath}")
            continue

        df = pd.read_parquet(fpath)
        if df.empty:
            logger.warning(f"Skipping {pair}: empty features: {fpath}")
            continue

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        if "close" not in df.columns:
            logger.warning(f"Skipping {pair}: close not found in features")
            continue

        actual = df.set_index("date")["close"].iloc[-args.history_days:]
        asof_ts = df["date"].iloc[-1]
        last_close = float(df["close"].iloc[-1])

        plt.figure(figsize=(13, 6))
        plt.plot(actual.index, actual.values, linewidth=2, label=f"{pair} actual close")

        for model in models:
            ppath = _predictions_path(cfg, args.frequency, model, pair, ingest_date)
            if not ppath.exists():
                logger.warning(f"No prediction file: {ppath}")
                continue

            pred = pd.read_parquet(ppath)
            if pred.empty:
                logger.warning(f"Empty prediction file: {ppath}")
                continue

            pred["pred_date"] = pd.to_datetime(pred["pred_date"])
            pred = pred.sort_values("pred_date")

            # jeśli plik zawiera kilka runów, weź najnowszy
            if "run_ts_utc" in pred.columns:
                latest_ts = pred["run_ts_utc"].max()
                pred = pred[pred["run_ts_utc"] == latest_ts]

            # oczekiwany schemat: pred_close_mean
            if "pred_close_mean" in pred.columns:
                y = pred["pred_close_mean"].astype(float).values
            else:
                # fallback: jeśli ktoś ma stare pliki z pred_return_1d – policzmy trajektorię
                if "pred_return_1d" in pred.columns:
                    rets = pred["pred_return_1d"].astype(float).values
                elif "pred_return" in pred.columns:
                    rets = pred["pred_return"].astype(float).values
                else:
                    logger.warning(f"Prediction missing close/return columns for model={model}, pair={pair}")
                    continue
                y = last_close * np.exp(np.cumsum(rets))

            x = pred["pred_date"].values
            plt.plot(x, y, linewidth=2, label=f"{model} pred close")

            # INTERWAŁY – tylko jeśli nie wyłączono flagą
            if not args.no_intervals:
                if "pred_close_lower" in pred.columns and "pred_close_upper" in pred.columns:
                    lo = pred["pred_close_lower"].astype(float).values
                    hi = pred["pred_close_upper"].astype(float).values
                    mask = np.isfinite(lo) & np.isfinite(hi)
                    if mask.any():
                        plt.fill_between(x[mask], lo[mask], hi[mask], alpha=0.15, label=f"{model} interval")

            summary_rows.append(
                {
                    "pair": pair,
                    "model": model,
                    "ingest_date": ingest_date,
                    "asof_date": str(asof_ts.date()),
                    "horizon_steps": int(pred["step"].max()) if "step" in pred.columns else len(pred),
                    "pred_close_t1": float(y[0]),
                    "pred_close_tH": float(y[-1]),
                }
            )

        plt.title(
            f"{pair} | actual close (last {args.history_days}d) + predicted close trajectories | ingest_date={ingest_date}",
            fontsize=12,
        )
        plt.xlabel("Date")
        plt.ylabel("Close")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        out_dir = out_root / f"pair={pair}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / ("forecast_vs_actual_no_interval.png" if args.no_intervals else "forecast_vs_actual.png")
        plt.savefig(out_path)
        plt.close()

        logger.info(f"Saved plot: {out_path}")

    if summary_rows:
        out_csv = out_root / "predictions_summary.csv"
        pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
        logger.info(f"Saved predictions summary: {out_csv}")


if __name__ == "__main__":
    main()

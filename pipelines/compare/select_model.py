from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from config.loader import load_config
from utils.logger import setup_logger
from pipelines.sarima.utils import _features_root, get_latest_ingest_date, list_pairs


def _metrics_dir(cfg: dict, frequency: str, model: str, pair: str, ingest_date: str) -> Path:
    metrics_root = Path(cfg.get("paths", {}).get("metrics", "data/metrics"))
    return (
        metrics_root
        / "backtest"
        / f"frequency={frequency}"
        / f"model={model}"
        / f"pair={pair}"
        / f"ingest_date={ingest_date}"
    )


def _load_metrics_one(cfg: dict, frequency: str, model: str, pair: str, ingest_date: str) -> dict | None:
    d = _metrics_dir(cfg, frequency, model, pair, ingest_date)

    pq = d / "metrics.parquet"
    js = d / "metrics.json"

    if pq.exists():
        df = pd.read_parquet(pq)
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    if js.exists():
        try:
            return json.loads(js.read_text())
        except Exception:
            return None

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frequency", default="daily")
    ap.add_argument("--ingest-date", default=None)
    ap.add_argument("--pairs", default="all", help="all or comma-separated e.g. EUR_USD,GBP_USD")
    ap.add_argument("--models", default="naive,sarima,prophet", help="comma-separated model ids")
    ap.add_argument("--metric", default="mae", help="mae|rmse|smape (lower is better)")
    args = ap.parse_args()

    cfg = load_config()

    # ingest_date bazujemy na FEATURES (to jest źródło prawdy dla dnia)
    features_root = _features_root(cfg, frequency=args.frequency)
    ingest_date = args.ingest_date or get_latest_ingest_date(features_root)

    if args.pairs == "all":
        pairs = list_pairs(cfg, features_root, ingest_date)
    else:
        pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    metric = args.metric.strip().lower()

    logger = setup_logger("compare.select_model", Path("logs/compare"))
    logger.info(
        f"Selecting best model for pairs={pairs} ingest_date={ingest_date} metric={metric} models={models}"
    )

    selections = []
    run_ts = datetime.now(timezone.utc).isoformat()

    # gdzie zapisujemy wybór (per pair)
    out_root = Path(cfg.get("paths", {}).get("models", "models")) / "active_models" / "forex_daily" / f"frequency={args.frequency}"
    out_root.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        candidates = []
        for model in models:
            m = _load_metrics_one(cfg, args.frequency, model, pair, ingest_date)
            if not m:
                continue
            if metric not in m or m.get(metric) is None:
                continue

            candidates.append(
                {
                    "pair": pair,
                    "model": model,
                    "metric": metric,
                    "metric_value": float(m[metric]),
                    "n": int(m.get("n", 0)),
                    "ingest_date": ingest_date,
                }
            )

        if not candidates:
            logger.warning(f"No candidates for pair={pair} (missing metrics?)")
            continue

        best = sorted(candidates, key=lambda x: x["metric_value"])[0]

        selection = {
            "pair": pair,
            "selected_model": best["model"],
            "metric": metric,
            "metric_value": best["metric_value"],
            "n": best["n"],
            "ingest_date": ingest_date,
            "selected_at": run_ts,
            "models_considered": models,
        }
        selections.append(selection)

        out_dir = out_root / f"pair={pair}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "active_model.json").write_text(json.dumps(selection, indent=2))

        logger.info(f"Selected for {pair}: {best['model']} {metric}={best['metric_value']:.6f}")

    if not selections:
        raise RuntimeError("No model selections computed.")

    # zapis zbiorczy
    summary_path = out_root / f"selections_ingest_date={ingest_date}.json"
    summary_path.write_text(json.dumps(selections, indent=2))
    logger.info(f"Model selection completed. Saved: {summary_path}")


if __name__ == "__main__":
    main()

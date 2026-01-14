# pipelines/sarima/train.py
from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from datetime import datetime, timezone
import argparse

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config.loader import load_config
from utils.logger import setup_logger
from pipelines.sarima.utils import list_pairs, _features_root, get_latest_ingest_date, load_return_series

MODELS_ROOT = Path("models/sarima")
LOGS_ROOT = Path("logs/sarima")

def _next_version(pair_dir: Path) -> str:
    versions = sorted([p.name for p in pair_dir.glob("v*") if p.is_dir()])
    return f"v{len(versions) + 1}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frequency", default="daily")
    ap.add_argument("--ingest-date", default=None)
    ap.add_argument("--pairs", default="all")
    args = ap.parse_args()

    cfg = load_config()
    root = _features_root(cfg, frequency=args.frequency)
    ingest_date = args.ingest_date or get_latest_ingest_date(root)

    if args.pairs == "all":
        pairs = list_pairs(cfg, root, ingest_date)
    else:
        pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]

    mcfg = cfg.get("models", {}).get("sarima", {}).get(args.frequency, {})
    order = tuple(mcfg.get("order", [1, 0, 1]))
    seasonal_order = tuple(mcfg.get("seasonal_order", [0, 0, 0, 0]))
    enforce_stationarity = bool(mcfg.get("enforce_stationarity", True))
    enforce_invertibility = bool(mcfg.get("enforce_invertibility", True))

    logger = setup_logger("sarima.train", LOGS_ROOT)
    logger.info(f"SARIMA train ingest_date={ingest_date}, pairs={pairs}, order={order}, seasonal_order={seasonal_order}")

    for pair in pairs:
        s, _ = load_return_series(pair, ingest_date=ingest_date, frequency=args.frequency)
        # safety
        y = s.values.astype(float)
        y = y[np.isfinite(y)]
        if len(y) < 80:
            logger.warning(f"Skipping {pair}: too short series len={len(y)}")
            continue

        model = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )
        result = model.fit(disp=False)

        pair_dir = MODELS_ROOT / pair
        pair_dir.mkdir(parents=True, exist_ok=True)

        version = _next_version(pair_dir)
        model_dir = pair_dir / version
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(result, f)

        metadata = {
            "model": "sarima",
            "pair": pair,
            "frequency": args.frequency,
            "ingest_date": ingest_date,
            "order": list(order),
            "seasonal_order": list(seasonal_order),
            "enforce_stationarity": enforce_stationarity,
            "enforce_invertibility": enforce_invertibility,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        # active_model.json (zamiast "latest" katalogu — ale możesz mieć oba)
        (pair_dir / "active_model.json").write_text(json.dumps({
            "active_version": version,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "ingest_date": ingest_date
        }, indent=2))

        # (opcjonalnie) utrzymuj też "latest" jako kopię
        latest_dir = pair_dir / "latest"
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        shutil.copytree(model_dir, latest_dir)

        logger.info(f"[{pair}] trained -> {version}")

    logger.info("SARIMA train done.")

if __name__ == "__main__":
    main()

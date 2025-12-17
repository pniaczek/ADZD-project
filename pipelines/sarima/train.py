import json
import pickle
from pathlib import Path
from datetime import datetime, UTC
import shutil

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pipelines.sarima.utils import load_series
from utils.logger import setup_logger


PAIR = "EUR_USD"
TRAIN_YEARS = 5   # kluczowe: dłuższe okno
MODELS_ROOT = Path("models/sarima")
LOGS_ROOT = Path("logs/sarima")


def main():
    logger = setup_logger(
        name="sarima.train",
        log_dir=LOGS_ROOT / PAIR,
    )

    logger.info(f"Starting SARIMA training for {PAIR}")

    # ======================
    # LOAD DATA
    # ======================
    series = load_series(PAIR).last(f"{TRAIN_YEARS}Y")
    logger.info(f"Loaded {len(series)} price observations")

    # --- log-returns ---
    returns = np.log(series).diff().dropna()
    logger.info(f"Using {len(returns)} log-return observations")

    # ======================
    # TRAIN SARIMA (CONSERVATIVE)
    # ======================
    model = SARIMAX(
        returns,
        order=(1, 0, 1),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=True,
        enforce_invertibility=True,
    )

    result = model.fit(disp=False)

    # ======================
    # VERSIONING
    # ======================
    pair_dir = MODELS_ROOT / PAIR
    pair_dir.mkdir(parents=True, exist_ok=True)

    versions = sorted(
        [p.name for p in pair_dir.glob("v*") if p.is_dir()]
    )
    next_version = f"v{len(versions) + 1}"

    model_dir = pair_dir / next_version
    model_dir.mkdir()

    # save model
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(result, f)

    metadata = {
        "model": "SARIMA",
        "pair": PAIR,
        "train_years": TRAIN_YEARS,
        "order": [1, 0, 1],
        "seasonal_order": [0, 0, 0, 0],
        "trained_at": datetime.now(UTC).isoformat(),
    }

    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )


    # ======================
    # UPDATE LATEST (SAFE)
    # ======================
    latest_dir = pair_dir / "latest"

    if latest_dir.exists() or latest_dir.is_symlink():
        if latest_dir.is_symlink():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)

    # COPY model version → latest
    shutil.copytree(model_dir, latest_dir)

    logger.info(f"Latest model updated → {model_dir.name}")



if __name__ == "__main__":
    main()

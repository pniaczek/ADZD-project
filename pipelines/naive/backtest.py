import json
from pathlib import Path
from datetime import datetime, UTC

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pipelines.sarima.utils import load_series
from utils.logger import setup_logger

PAIR = "EUR_USD"
BACKTEST_DAYS = 180

MODELS_ROOT = Path("models/naive")
LOGS_ROOT = Path("logs/naive")


def main():
    logger = setup_logger(
        name="naive.backtest",
        log_dir=LOGS_ROOT / PAIR,
    )

    logger.info(f"Starting NAIVE backtest for {PAIR}")

    # ======================
    # LOAD DATA
    # ======================
    series = load_series(PAIR)
    test = series[-BACKTEST_DAYS:]

    # ======================
    # NAIVE FORECAST
    # ======================
    df = pd.DataFrame({
        "y_true": test,
        "y_pred": test.shift(1),
    }).dropna()

    y_true = df["y_true"].values
    y_pred = df["y_pred"].values

    logger.info(f"Backtest observations after NaN filter: {len(y_true)}")

    # ======================
    # METRICS
    # ======================
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true))

    result = {
        "model": "NAIVE",
        "pair": PAIR,
        "backtest_days": BACKTEST_DAYS,
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "observations": int(len(y_true)),
        },
        "run_timestamp": datetime.now(UTC).isoformat(),
    }

    # ======================
    # SAVE BACKTEST
    # ======================
    model_dir = MODELS_ROOT / PAIR / "latest"
    model_dir.mkdir(parents=True, exist_ok=True)

    backtest_path = model_dir / "backtest.json"
    backtest_path.write_text(json.dumps(result, indent=2))

    logger.info("NAIVE backtest completed successfully")
    logger.info(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

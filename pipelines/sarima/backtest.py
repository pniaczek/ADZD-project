import json
import pickle
from pathlib import Path
from datetime import datetime, UTC

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pipelines.sarima.utils import load_series
from utils.logger import setup_logger

PAIR = "EUR_USD"
BACKTEST_DAYS = 180

MODELS_ROOT = Path("models/sarima")
LOGS_ROOT = Path("logs/sarima")


def main():
    logger = setup_logger(
        name="sarima.backtest",
        log_dir=LOGS_ROOT / PAIR,
    )

    logger.info(f"Starting SARIMA backtest for {PAIR}")

    # ======================
    # LOAD MODEL
    # ======================
    model_dir = MODELS_ROOT / PAIR / "latest"
    with open(model_dir / "model.pkl", "rb") as f:
        result = pickle.load(f)

    # ======================
    # LOAD DATA → LOG RETURNS
    # ======================
    prices = load_series(PAIR)
    returns = np.log(prices).diff().dropna()

    test = returns[-BACKTEST_DAYS:]

    # ======================
    # FORECAST RETURNS (ONE SHOT)
    # ======================
    fc = result.get_forecast(steps=len(test))
    y_pred = fc.predicted_mean.values
    y_true = test.values

    # safety
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    logger.info(f"Backtest observations: {len(y_true)}")

    # ======================
    # METRICS
    # ======================
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs(y_true - y_pred))

    backtest = {
        "model": "SARIMA",
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
    # SAVE
    # ======================
    out_path = model_dir / "backtest.json"
    out_path.write_text(json.dumps(backtest, indent=2))

    logger.info("SARIMA backtest completed")
    logger.info(json.dumps(backtest, indent=2))


if __name__ == "__main__":
    main()

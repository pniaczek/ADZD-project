import json
import pickle
from pathlib import Path
from datetime import datetime, UTC

import numpy as np
import pandas as pd

from pipelines.sarima.utils import load_series
from utils.logger import setup_logger


PAIR = "EUR_USD"
HORIZON_DAYS = 14

MODELS_ROOT = Path("models/sarima")
PREDICTIONS_ROOT = Path("data/predictions/sarima")


def main():
    logger = setup_logger(
        name="sarima.predict",
        log_dir=Path("logs/sarima") / PAIR,
    )

    # ======================
    # LOAD MODEL
    # ======================
    model_dir = MODELS_ROOT / PAIR / "latest"

    with open(model_dir / "model.pkl", "rb") as f:
        result = pickle.load(f)

    # ======================
    # LOAD DATA
    # ======================
    series = load_series(PAIR)
    last_price = series.iloc[-1]

    # ======================
    # FORECAST RETURNS
    # ======================
    fc = result.get_forecast(steps=HORIZON_DAYS)

    ret_mean = fc.predicted_mean
    conf = fc.conf_int(alpha=0.2)

    # ======================
    # RECONSTRUCT PRICE
    # ======================
    price_mean = last_price * np.exp(ret_mean.cumsum())
    price_lower = last_price * np.exp(conf.iloc[:, 0].cumsum())
    price_upper = last_price * np.exp(conf.iloc[:, 1].cumsum())

    df = pd.DataFrame({
        "mean": price_mean.values,
        "lower": price_lower.values,
        "upper": price_upper.values,
        "step": range(1, HORIZON_DAYS + 1),
        "pair": PAIR,
        "run_timestamp": datetime.now(UTC).isoformat(),
    })

    # ======================
    # SAVE
    # ======================
    run_date = datetime.now(UTC).date().isoformat()

    out_dir = (
        PREDICTIONS_ROOT
        / f"pair={PAIR}"
        / f"run_date={run_date}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "forecast.parquet"
    df.to_parquet(out_path, index=False)

    logger.info(f"SARIMA forecast saved to {out_path}")
    logger.info(f"Forecast horizon: {HORIZON_DAYS} days")


if __name__ == "__main__":
    main()

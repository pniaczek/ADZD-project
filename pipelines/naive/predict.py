import pandas as pd
from pathlib import Path
from datetime import datetime, UTC

from pipelines.sarima.utils import load_series
from utils.logger import setup_logger

PAIR = "EUR_USD"
HORIZON = 10

def main():
    logger = setup_logger(
        name="naive.predict",
        log_dir=Path("logs/naive") / PAIR,
    )

    logger.info(f"Running NAIVE forecast for {PAIR}")

    series = load_series(PAIR)
    last_value = series.iloc[-1]

    forecast = pd.DataFrame({
        "step": range(1, HORIZON + 1),
        "mean": [last_value] * HORIZON,
        "lower": [last_value] * HORIZON,
        "upper": [last_value] * HORIZON,
        "pair": PAIR,
        "run_timestamp": datetime.now(UTC).isoformat(),
    })

    run_date = datetime.now(UTC).date().isoformat()
    output_path = (
        Path("data/predictions/naive")
        / f"pair={PAIR}"
        / f"run_date={run_date}"
    )

    output_path.mkdir(parents=True, exist_ok=True)
    forecast.to_parquet(output_path / "forecast.parquet", index=False)

    logger.info(f"NAIVE forecast saved to {output_path}")

if __name__ == "__main__":
    main()

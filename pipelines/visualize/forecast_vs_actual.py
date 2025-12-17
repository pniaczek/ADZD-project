from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

from pipelines.sarima.utils import load_series
from utils.logger import setup_logger


PAIR = "EUR_USD"
HISTORY_DAYS = 120

REPORTS_ROOT = Path("reports/figures")
PREDICTIONS_ROOT = Path("data/predictions")
MODELS_ROOT = Path("models")


def load_forecast(model: str, pair: str, run_date: str, logger):
    """
    Load forecast parquet for given model.
    Returns DataFrame or None if missing.
    """
    path = (
        PREDICTIONS_ROOT
        / model
        / f"pair={pair}"
        / f"run_date={run_date}"
        / "forecast.parquet"
    )

    if not path.exists():
        logger.warning(f"No forecast found for {model.upper()} at {path}")
        return None

    df = pd.read_parquet(path)
    return df


def attach_forecast_dates(forecast: pd.DataFrame, last_actual_date):
    """
    Attach datetime index to forecast based on last actual date.
    """
    idx = pd.date_range(
        start=last_actual_date,
        periods=len(forecast) + 1,
        freq="D",
    )[1:]

    forecast = forecast.copy()
    forecast["date"] = idx
    return forecast.set_index("date")


def main():
    logger = setup_logger(
        name="visualize.forecast_vs_actual",
        log_dir=Path("logs/visualize") / PAIR,
    )

    # ======================
    # LOAD ACTIVE MODEL
    # ======================
    active_path = MODELS_ROOT / "active_model.json"
    active = json.loads(active_path.read_text())

    winner = active["selected_model"]
    run_id = active["selected_at"]
    run_date = run_id[:10]

    logger.info(
        f"Active model: {winner}, run_id={run_id}"
    )

    # ======================
    # LOAD ACTUAL DATA
    # ======================
    series = load_series(PAIR)
    actual = series[-HISTORY_DAYS:]

    last_actual_date = actual.index[-1]

    # ======================
    # LOAD FORECASTS
    # ======================
    forecasts = {}

    for model in ["sarima", "naive"]:
        df = load_forecast(model, PAIR, run_date, logger)
        if df is not None:
            forecasts[model] = attach_forecast_dates(df, last_actual_date)

    if not forecasts:
        logger.error("No forecasts available – aborting visualization")
        return

    # ======================
    # OUTPUT DIR
    # ======================
    out_dir = (
        REPORTS_ROOT
        / PAIR
        / f"run_id={run_id}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ======================
    # PLOT
    # ======================
    plt.figure(figsize=(13, 6))

    # Actual
    plt.plot(
        actual.index,
        actual.values,
        label="Actual",
        linewidth=2,
        color="black",
    )

    # SARIMA
    if "sarima" in forecasts:
        f = forecasts["sarima"]
        plt.plot(
            f.index,
            f["mean"],
            label="SARIMA forecast",
            linewidth=2,
        )

        if {"lower", "upper"}.issubset(f.columns):
            plt.fill_between(
                f.index,
                f["lower"],
                f["upper"],
                alpha=0.25,
                label="SARIMA confidence interval",
            )

    # NAIVE
    if "naive" in forecasts:
        f = forecasts["naive"]
        plt.plot(
            f.index,
            f["mean"],
            label="NAIVE forecast",
            linestyle="--",
            linewidth=2,
        )

    plt.title(
        f"{PAIR} | forecast vs actual | winner={winner}",
        fontsize=12,
    )
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    out_path = out_dir / "forecast_vs_actual.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    logger.info(f"Saved forecast plot to {out_path}")


if __name__ == "__main__":
    main()

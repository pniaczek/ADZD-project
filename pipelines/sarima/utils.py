import pandas as pd
from pathlib import Path

def load_series(pair: str) -> pd.Series:
    path = Path("data/processed/forex_daily") / f"pair={pair}"
    df = pd.read_parquet(path)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    series = df.set_index("date")["close"]
    series = series.asfreq("D")  # ważne dla SARIMA

    return series

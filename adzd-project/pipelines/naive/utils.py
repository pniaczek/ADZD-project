# pipelines/naive/utils.py
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/forex_daily")

def load_series(pair: str) -> pd.Series:
    df = pd.read_parquet(DATA_PATH / f"pair={pair}")
    df = df.sort_values("date")
    return df.set_index("date")["close"]

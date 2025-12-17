import requests
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit

from config.loader import load_config
from pathlib import Path
from utils.logger import setup_logger



def fetch_alpha_vantage(cfg: dict) -> dict:
    av = cfg["alpha_vantage"]

    params = {
        "function": av["function"],
        "from_symbol": av["from_symbol"],
        "to_symbol": av["to_symbol"],
        "apikey": av["api_key"],
        "outputsize": av["output_size"],
    }

    resp = requests.get(
        "https://www.alphavantage.co/query",
        params=params,
        headers={"User-Agent": "alpha-spark-ingest/1.0"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    logger = setup_logger(
    name="spark.ingest.alpha_vantage",
    log_dir=Path("logs/spark"),
    )

    cfg = load_config()
    av = cfg["alpha_vantage"]

    # SparkSession – BEZ mastera
    spark = (
        SparkSession.builder
        .appName("alpha-vantage-fx-daily-ingest")
        .getOrCreate()
    )

    raw_json = fetch_alpha_vantage(cfg)

    ts_key = "Time Series FX (Daily)"
    if ts_key not in raw_json:
        raise RuntimeError(
            f"Brak klucza '{ts_key}' w odpowiedzi API: {raw_json}"
        )

    rows = []
    for ts, values in raw_json[ts_key].items():
        rows.append({
            "date": ts,
            "open": float(values["1. open"]),
            "high": float(values["2. high"]),
            "low": float(values["3. low"]),
            "close": float(values["4. close"]),
        })

    df = spark.createDataFrame(rows)

    df = (
        df
        .withColumn("from_symbol", lit(av["from_symbol"]))
        .withColumn("to_symbol", lit(av["to_symbol"]))
        .withColumn("frequency", lit("daily"))
        .withColumn("ingest_date", lit(datetime.utcnow().date().isoformat()))
    )

    output_path = (
        f"{cfg['paths']['raw']}/alpha_vantage/"
        f"market=forex/"
        f"pair={av['from_symbol']}_{av['to_symbol']}/"
        f"frequency=daily/"
        f"ingest_date={datetime.utcnow().date().isoformat()}"
    )

    (
        df
        .orderBy(col("date"))
        .write
        .mode("overwrite")
        .parquet(output_path)
    )

    logger.info(f"Dane zapisane do: {output_path}")
    logger.info(f"Liczba rekordów: {df.count()}")


    spark.stop()


if __name__ == "__main__":
    main()

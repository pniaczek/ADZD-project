import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple

from pyspark import StorageLevel
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

from config.loader import load_config
from utils.logger import setup_logger


AV_URL = "https://www.alphavantage.co/query"


def fetch_alpha_vantage_pair(av_cfg: dict, from_symbol: str, to_symbol: str) -> dict:
    params = {
        "function": av_cfg["function"],
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "apikey": av_cfg["api_key"],
        "outputsize": av_cfg.get("output_size", "compact"),
    }

    resp = requests.get(
        AV_URL,
        params=params,
        headers={"User-Agent": "alpha-spark-ingest/1.0"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def parse_fx_daily(raw_json: dict) -> List[Dict[str, Any]]:
    ts_key = "Time Series FX (Daily)"
    if ts_key not in raw_json:
        raise RuntimeError(
            f"Missing '{ts_key}'. Response keys={list(raw_json.keys())}, raw={raw_json}"
        )

    rows = []
    for ts, values in raw_json[ts_key].items():
        rows.append(
            {
                "date": ts,
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
            }
        )
    return rows


def fetch_with_retry(av_cfg: dict, from_symbol: str, to_symbol: str, max_retries: int = 5) -> dict:
    backoff = 2.0
    last_err = None

    for _ in range(1, max_retries + 1):
        try:
            data = fetch_alpha_vantage_pair(av_cfg, from_symbol, to_symbol)

            # Rate limit – AlphaVantage używa różnych kluczy: Note / Information
            if "Note" in data:
                raise RuntimeError(f"Rate limit hit: {data['Note']}")
            if "Information" in data and "Thank you for using Alpha Vantage" in str(data["Information"]):
                raise RuntimeError(f"Rate limit hit: {data['Information']}")

            return data

        except Exception as e:
            last_err = e
            time.sleep(backoff)
            backoff *= 1.8

    raise RuntimeError(f"Failed after {max_retries} retries for {from_symbol}/{to_symbol}: {last_err}")


def _ingest_partition(
    pairs: Iterable[Tuple[str, str]],
    av_cfg: dict,
    ingest_date: str,
    frequency: str,
    min_sleep: float,
) -> Iterable[Row]:
    """
    Funkcja działa w executorze dla jednej partycji par walutowych.
    Zwraca wiersze Row, które później Spark sklei w DataFrame.
    """
    out: List[Row] = []

    for (fs, ts) in pairs:
        try:
            raw = fetch_with_retry(av_cfg, fs, ts, max_retries=int(av_cfg.get("max_retries", 5)))
            rows = parse_fx_daily(raw)
        except Exception as e:
            out.append(
                Row(
                    date=None,
                    open=None,
                    high=None,
                    low=None,
                    close=None,
                    from_symbol=fs,
                    to_symbol=ts,
                    pair=f"{fs}_{ts}",
                    frequency=frequency,
                    ingest_date=ingest_date,
                    error=str(e),
                )
            )
            # po błędzie też śpimy (nie dobijamy limitów)
            time.sleep(min_sleep)
            continue

        for r in rows:
            out.append(
                Row(
                    date=r["date"],
                    open=r["open"],
                    high=r["high"],
                    low=r["low"],
                    close=r["close"],
                    from_symbol=fs,
                    to_symbol=ts,
                    pair=f"{fs}_{ts}",
                    frequency=frequency,
                    ingest_date=ingest_date,
                    error=None,
                )
            )

        # ~1 request / 1s (free key)
        time.sleep(min_sleep)

    return out


def build_spark(cfg: dict) -> SparkSession:
    env = cfg.get("project", {}).get("environment", "local")
    master = cfg.get("spark", {}).get("master", {}).get(env)

    spark_builder = SparkSession.builder.appName("alpha-vantage-fx-daily-ingest")
    if master:
        spark_builder = spark_builder.master(master)

    event_cfg = cfg.get("spark", {}).get("event_log", {}) or {}
    if event_cfg.get("enabled"):
        event_dir = event_cfg.get("dir")
        if event_dir:
            event_dir = str(Path(event_dir).absolute())
            spark_builder = (
                spark_builder.config("spark.eventLog.enabled", "true")
                .config("spark.eventLog.dir", event_dir)
            )

    return spark_builder.getOrCreate()


def main():
    logger = setup_logger(
        name="spark.ingest.alpha_vantage",
        log_dir=Path("logs/spark"),
    )

    cfg = load_config()
    av_cfg = cfg["alpha_vantage"]

    pairs = av_cfg.get("pairs")
    if not pairs:
        pairs = [[av_cfg["from_symbol"], av_cfg["to_symbol"]]]
    pairs = [(p[0], p[1]) for p in pairs]

    ingest_date = datetime.utcnow().date().isoformat()
    frequency = "daily"

    spark = build_spark(cfg)

    # Rate limit i równoległość
    min_sleep = float(av_cfg.get("min_seconds_between_requests", 1.1))
    max_parallel = int(av_cfg.get("max_parallel_pairs", 1))
    parallelism = min(max_parallel, len(pairs))
    parallelism = max(1, parallelism)

    logger.info(
        f"Ingest start. pairs={pairs}, parallelism={parallelism}, ingest_date={ingest_date}, min_sleep={min_sleep}"
    )

    schema = StructType([
        StructField("date", StringType(), True),
        StructField("open", DoubleType(), True),
        StructField("high", DoubleType(), True),
        StructField("low", DoubleType(), True),
        StructField("close", DoubleType(), True),
        StructField("from_symbol", StringType(), True),
        StructField("to_symbol", StringType(), True),
        StructField("pair", StringType(), True),
        StructField("frequency", StringType(), True),
        StructField("ingest_date", StringType(), True),
        StructField("error", StringType(), True),
    ])

    sc = spark.sparkContext
    pairs_rdd = sc.parallelize(pairs, parallelism)

    rows_rdd = pairs_rdd.mapPartitions(
        lambda it: _ingest_partition(
            it,
            av_cfg=av_cfg,
            ingest_date=ingest_date,
            frequency=frequency,
            min_sleep=min_sleep,
        )
    )

    # KLUCZ: materializacja tylko raz (żeby nie odpalać ponownie requestów przy count/write)
    df = spark.createDataFrame(rows_rdd, schema=schema).persist(StorageLevel.MEMORY_AND_DISK)
    _ = df.count()  # materializuj (jedna akcja)

    df_ok = df.filter(col("date").isNotNull())
    df_err = df.filter(col("date").isNull()).filter(col("error").isNotNull())

    ok_count = df_ok.count()
    err_count = df_err.count()

    if ok_count == 0:
        logger.warning("Brak poprawnych rekordów (ok_count=0). Prawdopodobnie rate limit. Nie zapisuję danych OK.")
    else:
        output_root = f"{cfg['paths']['raw']}/alpha_vantage/market=forex/frequency={frequency}"
        (
            df_ok.write
                .mode("overwrite")
                .partitionBy("pair", "ingest_date")
                .parquet(output_root)
        )
        logger.info(f"Dane OK zapisane do: {output_root} (partitionBy pair, ingest_date)")

    # błędy zawsze warto zapisać (diagnostyka/monitoring)
    if err_count > 0:
        err_root = (
            f"{cfg['paths']['raw']}/alpha_vantage_errors/"
            f"market=forex/frequency={frequency}/ingest_date={ingest_date}"
        )
        df_err.write.mode("overwrite").parquet(err_root)
        logger.warning(f"Zapisano błędy ingestu: {err_count} do {err_root}")

    logger.info(f"Liczba rekordów OK: {ok_count}")
    logger.info(f"Liczba rekordów błędów: {err_count}")

    spark.stop()


if __name__ == "__main__":
    main()

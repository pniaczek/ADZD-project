from pathlib import Path
import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lag,
    lead,
    log,
    datediff,
    lit,
    max as spark_max,
    count as spark_count,
    avg,
    stddev_samp,
)
from pyspark.sql.window import Window
from pyspark.sql.types import DateType

from config.loader import load_config
from utils.logger import setup_logger


def build_spark(cfg: dict) -> SparkSession:
    env = cfg.get("project", {}).get("environment", "local")
    master = cfg.get("spark", {}).get("master", {}).get(env)

    spark_builder = SparkSession.builder.appName("forex-daily-preprocess")

    if master:
        spark_builder = spark_builder.master(master)

    event_cfg = cfg.get("spark", {}).get("event_log", {}) or {}
    if event_cfg.get("enabled") and event_cfg.get("dir"):
        spark_builder = spark_builder.config("spark.eventLog.enabled", "true").config(
            "spark.eventLog.dir", str(Path(event_cfg.get("dir")).absolute())
        )

    spark = spark_builder.getOrCreate()

    # Produkcyjnie: nadpisuj tylko partycje obecne w zapisie (idempotentny rerun)
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    return spark


def parse_args():
    p = argparse.ArgumentParser(description="Forex daily preprocessing + feature engineering (Spark)")
    p.add_argument("--frequency", default="daily", choices=["daily"])
    p.add_argument(
        "--ingest-date",
        default="latest",
        help="Ingest date to process (YYYY-MM-DD) or 'latest' (default)",
    )
    p.add_argument(
        "--allow-any-gap",
        action="store_true",
        help="If set, do not filter on gap_days (default filters to {1,3} for FX)",
    )
    p.add_argument(
        "--output-root",
        default=None,
        help="Override output root (otherwise uses cfg.paths.features or data/features)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    logger = setup_logger(
        name="spark.preprocess.forex_daily",
        log_dir=Path("logs/spark"),
    )

    cfg = load_config()
    spark = build_spark(cfg)

    frequency = args.frequency

    # RAW (layout po A)
    input_root = f"{cfg['paths']['raw']}/alpha_vantage/market=forex/frequency={frequency}"

    # FEATURES output
    default_features_root = f"{cfg['paths'].get('features', 'data/features')}/forex_daily/frequency={frequency}"
    output_root = args.output_root or default_features_root

    logger.info(f"Reading raw from: {input_root}")

    # basePath pomaga Sparkowi poprawnie traktować partycje
    df = spark.read.option("basePath", input_root).parquet(input_root)

    # Wybór ingest_date
    if args.ingest_date == "latest":
        latest_ingest_date = (
            df.select(spark_max(col("ingest_date")).alias("max_ingest"))
              .collect()[0]["max_ingest"]
        )
        ingest_date = latest_ingest_date
    else:
        ingest_date = args.ingest_date

    logger.info(f"Processing ingest_date: {ingest_date}")
    df = df.filter(col("ingest_date") == lit(ingest_date))

    # Sanity / typy / duplikaty
    df = (
        df
        .withColumn("date", col("date").cast(DateType()))
        .dropDuplicates(["pair", "date"])
        .filter(
            (col("open") > 0)
            & (col("high") >= col("low"))
            & (col("close") > 0)
        )
    )

    # Window per pair (kluczowe)
    w = Window.partitionBy("pair").orderBy("date")

    # Rolling windows: historia bez bieżącej obserwacji
    w5 = w.rowsBetween(-5, -1)
    w20 = w.rowsBetween(-20, -1)

    df = (
        df
        .withColumn("prev_date", lag("date").over(w))
        .withColumn("prev_close", lag("close").over(w))
        .withColumn("gap_days", datediff(col("date"), col("prev_date")))
        .withColumn("log_return", log(col("close") / col("prev_close")))
        .withColumn("range", col("high") - col("low"))
        .withColumn("body", col("close") - col("open"))

        # Lagi zwrotu
        .withColumn("lag_ret_1", lag("log_return", 1).over(w))
        .withColumn("lag_ret_2", lag("log_return", 2).over(w))
        .withColumn("lag_ret_5", lag("log_return", 5).over(w))

        # Rolling mean/std
        .withColumn("roll_mean_5", avg(col("log_return")).over(w5))
        .withColumn("roll_mean_20", avg(col("log_return")).over(w20))
        .withColumn("roll_std_5", stddev_samp(col("log_return")).over(w5))
        .withColumn("roll_std_20", stddev_samp(col("log_return")).over(w20))
    )

    # Target: zwrot jutra
    df = df.withColumn("target_return_1d", lead("log_return", 1).over(w))

    # DropNA dla cech/targetu
    df = df.dropna(subset=[
        "log_return", "target_return_1d",
        "lag_ret_1", "lag_ret_2", "lag_ret_5",
        "roll_mean_5", "roll_std_5",
        "roll_mean_20", "roll_std_20",
        "gap_days"
    ])

    # Gap policy (FX: 1 dzień lub weekend 3 dni)
    if not args.allow_any_gap:
        df = df.filter(col("gap_days").isin([1, 3]))

    # Przypnij ingest_date do outputu (spójny kontrakt partycji)
    df = df.withColumn("ingest_date", lit(ingest_date))

    # Lepsza lokalność zapisu per pair
    df = df.repartition(col("pair"))

    logger.info(f"Writing features to: {output_root} (partitionBy pair, ingest_date)")
    (
        df.write
        .mode("overwrite")
        .partitionBy("pair", "ingest_date")
        .parquet(output_root)
    )

    # Metryka końcowa (jedna, jawna)
    recs = df.select(spark_count(lit(1)).alias("n")).collect()[0]["n"]
    logger.info(f"Preprocessing done. ingest_date={ingest_date}, records={recs}")

    spark.stop()


if __name__ == "__main__":
    main()

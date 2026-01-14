from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lit,
    max as spark_max,
    count as spark_count,
    date_add,
    lag,
)
from pyspark.sql.window import Window

from config.loader import load_config
from utils.logger import setup_logger


def build_spark(cfg: dict) -> SparkSession:
    env = cfg.get("project", {}).get("environment", "local")
    master = cfg.get("spark", {}).get("master", {}).get(env)

    spark_builder = SparkSession.builder.appName("forex-daily-predict")

    if master:
        spark_builder = spark_builder.master(master)

    event_cfg = cfg.get("spark", {}).get("event_log", {}) or {}
    if event_cfg.get("enabled") and event_cfg.get("dir"):
        spark_builder = spark_builder.config("spark.eventLog.enabled", "true").config(
            "spark.eventLog.dir", str(Path(event_cfg.get("dir")).absolute())
        )

    spark = spark_builder.getOrCreate()

    # Idempotentny overwrite tylko partycji obecnych w zapisie
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    return spark


def parse_args():
    p = argparse.ArgumentParser(description="Forex daily predict (Spark)")
    p.add_argument("--frequency", default="daily", choices=["daily"])
    p.add_argument("--ingest-date", default="latest", help="YYYY-MM-DD or 'latest'")
    p.add_argument("--model", default="naive", choices=["naive", "seasonal_naive"])
    p.add_argument("--seasonal-lag", type=int, default=5, help="Lag for seasonal_naive (default 5)")
    return p.parse_args()


def main():
    args = parse_args()

    logger = setup_logger(
        name="spark.predict.forex_daily",
        log_dir=Path("logs/spark"),
    )

    cfg = load_config()
    spark = build_spark(cfg)

    frequency = args.frequency
    model_name = args.model
    run_ts_utc = datetime.utcnow().isoformat()

    # Wejście: features z B
    features_root = f"{cfg['paths'].get('features', 'data/features')}/forex_daily/frequency={frequency}"

    # Wyjście: predictions
    preds_root = f"{cfg['paths'].get('predictions', 'data/predictions')}/forex_daily/frequency={frequency}/model={model_name}"

    logger.info(f"Reading features from: {features_root}")

    df = spark.read.parquet(features_root)

    # wybór ingest_date
    if args.ingest_date == "latest":
        ingest_date = df.select(spark_max(col("ingest_date")).alias("max_ingest")).collect()[0]["max_ingest"]
    else:
        ingest_date = args.ingest_date

    logger.info(f"Predict model={model_name}, ingest_date={ingest_date}, frequency={frequency}")
    df = df.filter(col("ingest_date") == lit(ingest_date))

    # Minimalny kontrakt wejścia
    required_cols = {"pair", "date", "log_return", "ingest_date"}
    missing = required_cols.difference(set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns in features dataset: {missing}")

    # Okno per para
    w = Window.partitionBy("pair").orderBy("date")

    # Predykcja:
    # - naive: jutro ~ dzisiaj => pred_return_1d = log_return(t)
    # - seasonal_naive: jutro ~ wartość sprzed seasonal_lag dni => pred_return_1d = lag(log_return, seasonal_lag)
    if model_name == "naive":
        df_pred_all = df.select("pair", "date", "ingest_date", "log_return").withColumn(
            "pred_return_1d", col("log_return")
        )
    else:
        df_pred_all = (
            df.select("pair", "date", "ingest_date", "log_return")
              .withColumn("pred_return_1d", lag(col("log_return"), args.seasonal_lag).over(w))
        )

    # Bierzemy najnowszy dostępny wiersz per para (as-of) i tworzymy predykcję na następny dzień
    df_latest = (
        df_pred_all
        .withColumn("max_date", spark_max(col("date")).over(Window.partitionBy("pair")))
        .filter(col("date") == col("max_date"))
        .drop("max_date")
    )

    # Usuń pary, dla których pred_return_1d nie da się policzyć (np. za krótka historia w seasonal)
    df_latest = df_latest.dropna(subset=["pred_return_1d"])

    # pred_date = asof_date + 1
    df_out = (
        df_latest
        .withColumnRenamed("date", "asof_date")
        .withColumn("pred_date", date_add(col("asof_date"), 1))
        .withColumn("model", lit(model_name))
        .withColumn("frequency", lit(frequency))
        .withColumn("run_ts_utc", lit(run_ts_utc))
    )

    n_pairs = df_out.select("pair").distinct().count()
    logger.info(f"Predictions prepared for pairs: {n_pairs}")

    if n_pairs == 0:
        logger.warning("No predictions produced (empty). Skipping write.")
        spark.stop()
        return

    logger.info(f"Writing predictions to: {preds_root} (partitionBy pair, ingest_date)")
    (
        df_out.write
        .mode("overwrite")
        .partitionBy("pair", "ingest_date")
        .parquet(preds_root)
    )

    logger.info("Predict done.")
    spark.stop()


if __name__ == "__main__":
    main()

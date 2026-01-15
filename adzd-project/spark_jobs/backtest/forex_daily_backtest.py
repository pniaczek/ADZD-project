from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lit,
    lag,
    max as spark_max,
    abs as spark_abs,
    sqrt,
    avg,
    count as spark_count,
    when,
)
from pyspark.sql.window import Window

from config.loader import load_config
from utils.logger import setup_logger


EPS = 1e-12


def build_spark(cfg: dict) -> SparkSession:
    env = cfg.get("project", {}).get("environment", "local")
    master = cfg.get("spark", {}).get("master", {}).get(env)

    spark_builder = SparkSession.builder.appName("forex-daily-backtest")

    if master:
        spark_builder = spark_builder.master(master)

    event_cfg = cfg.get("spark", {}).get("event_log", {}) or {}
    if event_cfg.get("enabled") and event_cfg.get("dir"):
        spark_builder = spark_builder.config("spark.eventLog.enabled", "true").config(
            "spark.eventLog.dir", str(Path(event_cfg.get("dir")).absolute())
        )

    spark = spark_builder.getOrCreate()

    # Idempotentny overwrite tylko partycji w zapisie (produkcyjnie)
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    return spark


def parse_args():
    p = argparse.ArgumentParser(description="Forex daily backtest (Spark)")
    p.add_argument("--frequency", default="daily", choices=["daily"])
    p.add_argument("--ingest-date", default="latest", help="YYYY-MM-DD or 'latest'")
    p.add_argument("--model", default="naive", choices=["naive", "seasonal_naive"])
    p.add_argument(
        "--seasonal-lag",
        type=int,
        default=5,
        help="Lag for seasonal_naive (default 5 for FX weekdays)",
    )
    p.add_argument("--write-preds", action="store_true", help="Write per-row predictions parquet")
    return p.parse_args()


def main():
    args = parse_args()

    logger = setup_logger(
        name="spark.backtest.forex_daily",
        log_dir=Path("logs/spark"),
    )

    cfg = load_config()
    spark = build_spark(cfg)

    frequency = args.frequency
    model_name = args.model
    run_ts_utc = datetime.utcnow().isoformat()

    # FEATURES input (z B)
    features_root = f"{cfg['paths'].get('features', 'data/features')}/forex_daily/frequency={frequency}"
    # OUTPUT: metrics
    metrics_root = f"{cfg['paths'].get('metrics', 'data/metrics')}/backtest/frequency={frequency}/model={model_name}"
    # OUTPUT: preds (opcjonalne)
    preds_root = f"{cfg['paths'].get('backtest_preds', 'data/backtest_preds')}/frequency={frequency}/model={model_name}"

    logger.info(f"Reading features from: {features_root}")
    df = spark.read.parquet(features_root)

    # wybór ingest_date
    if args.ingest_date == "latest":
        ingest_date = df.select(spark_max(col("ingest_date")).alias("max_ingest")).collect()[0]["max_ingest"]
    else:
        ingest_date = args.ingest_date

    logger.info(f"Backtest model={model_name}, ingest_date={ingest_date}, frequency={frequency}")

    df = df.filter(col("ingest_date") == lit(ingest_date))

    # Minimalny kontrakt wejścia
    required_cols = {"pair", "date", "log_return", "target_return_1d", "ingest_date"}
    missing = required_cols.difference(set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns in features dataset: {missing}")

    # Window per pair
    w = Window.partitionBy("pair").orderBy("date")

    # Predykcja baseline
    if model_name == "naive":
        # jutro ~ dzisiaj: y_pred = log_return(t)
        # ponieważ target_return_1d to lead(log_return,1),
        # to predykcję dla targetu w wierszu t możemy ustawić jako lag(lead?) albo po prostu lag(target)?,
        # ale najczytelniej: y_pred = lag(target_return_1d, 1) == log_return(t)
        y_pred = lag(col("target_return_1d"), 1).over(w)
    else:
        # seasonal naive: jutro ~ wartość sprzed 'seasonal_lag' dni
        # czyli y_pred = lag(target_return_1d, seasonal_lag) => approx log_return(t - seasonal_lag + 1)
        y_pred = lag(col("target_return_1d"), args.seasonal_lag).over(w)

    df_pred = (
        df.select("pair", "date", "ingest_date", "log_return", "target_return_1d")
          .withColumn("y_true", col("target_return_1d"))
          .withColumn("y_pred", y_pred)
          .dropna(subset=["y_true", "y_pred"])
    )

    # Metryki per wiersz
    df_pred = (
        df_pred
        .withColumn("abs_err", spark_abs(col("y_true") - col("y_pred")))
        .withColumn("sq_err", (col("y_true") - col("y_pred")) * (col("y_true") - col("y_pred")))
        .withColumn(
            "smape",
            (lit(2.0) * col("abs_err")) /
            (spark_abs(col("y_true")) + spark_abs(col("y_pred")) + lit(EPS))
        )

    )

    # Agregacja metryk per pair
    metrics_df = (
        df_pred.groupBy("pair", "ingest_date")
        .agg(
            spark_count(lit(1)).alias("n"),
            avg(col("abs_err")).alias("mae"),
            sqrt(avg(col("sq_err"))).alias("rmse"),
            avg(col("smape")).alias("smape"),
        )
        .withColumn("model", lit(model_name))
        .withColumn("frequency", lit(frequency))
        .withColumn("run_ts_utc", lit(run_ts_utc))
    )

    total_pairs = metrics_df.count()
    logger.info(f"Computed metrics for pairs: {total_pairs}")

    if total_pairs == 0:
        logger.warning("No metrics produced (empty). Skipping writes.")
        spark.stop()
        return

    # Zapis metryk (produkcyjnie: overwrite tylko partycji obecnych w zapisie)
    logger.info(f"Writing metrics to: {metrics_root} (partitionBy pair, ingest_date)")
    (
        metrics_df.write
        .mode("overwrite")
        .partitionBy("pair", "ingest_date")
        .parquet(metrics_root)
    )

    # Zapis predykcji (opcjonalnie)
    if args.write_preds:
        logger.info(f"Writing backtest predictions to: {preds_root} (partitionBy pair, ingest_date)")
        (
            df_pred
            .withColumn("model", lit(model_name))
            .withColumn("frequency", lit(frequency))
            .withColumn("run_ts_utc", lit(run_ts_utc))
            .write
            .mode("overwrite")
            .partitionBy("pair", "ingest_date")
            .parquet(preds_root)
        )

    # Krótki log metryk (bez dodatkowych akcji na df_pred)
    sample = metrics_df.orderBy(col("rmse").desc()).limit(5).collect()
    for r in sample:
        logger.info(
            f"[TOP RMSE] pair={r['pair']} n={r['n']} mae={r['mae']:.6f} rmse={r['rmse']:.6f} smape={r['smape']:.6f}"
        )

    logger.info("Backtest done.")
    spark.stop()


if __name__ == "__main__":
    main()

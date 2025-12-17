from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lag, lead, log, datediff
)
from pyspark.sql.window import Window
from pyspark.sql.types import DateType
from pathlib import Path
from utils.logger import setup_logger



def main():
    logger = setup_logger(
    name="spark.preprocess.forex_daily",
    log_dir=Path("logs/spark"),
    )

    spark = (
        SparkSession.builder
        .appName("forex-daily-preprocess")
        .getOrCreate()
    )

    input_path = "data/raw/alpha_vantage/market=forex/pair=EUR_USD/frequency=daily/*"
    output_path = "data/processed/forex_daily/pair=EUR_USD"

    df = spark.read.parquet(input_path)

    df = (
        df
        .withColumn("date", col("date").cast(DateType()))
        .dropDuplicates(["date"])
        .filter(
            (col("open") > 0) &
            (col("high") >= col("low")) &
            (col("close") > 0)
        )
    )

    w = Window.orderBy("date")

    df = (
        df
        .withColumn("prev_close", lag("close").over(w))
        .withColumn("log_return", log(col("close") / col("prev_close")))
        .withColumn("gap_days", datediff(col("date"), lag("date").over(w)))
        .withColumn("range", col("high") - col("low"))
        .withColumn("body", col("close") - col("open"))
        .withColumn("target_return_1d", lead("log_return", 1).over(w))
    )

    df = df.dropna(subset=["log_return", "target_return_1d"])

    (
        df
        .orderBy("date")
        .write
        .mode("overwrite")
        .parquet(output_path)
    )

    logger.info(f"Preprocessing zapisany do: {output_path}")
    logger.info(f"Liczba rekordów po preprocessingu: {df.count()}")


    spark.stop()


if __name__ == "__main__":
    main()

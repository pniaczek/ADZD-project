from pyspark.sql import SparkSession

def create_spark_session(cfg):
    spark_cfg = cfg["spark"]

    spark = (
        SparkSession.builder
        .appName(spark_cfg["app_name"])
        .master(spark_cfg["master"][cfg["project"]["environment"]])
        .config(
            "spark.eventLog.enabled",
            str(spark_cfg["event_log"]["enabled"]).lower()
        )
        .config(
            "spark.eventLog.dir",
            spark_cfg["event_log"]["dir"]
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel(spark_cfg.get("log_level", "INFO"))
    return spark

from config.loader import load_config
from spark_jobs.utils.spark_session import create_spark_session
from spark_jobs.ingest.alpha_vantage import AlphaVantageIngestor

def main():
    cfg = load_config()
    spark = create_spark_session(cfg)

    ingestor = AlphaVantageIngestor(cfg)
    ingestor.run(spark)

    spark.stop()

if __name__ == "__main__":
    main()

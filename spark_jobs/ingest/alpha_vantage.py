import requests
from pyspark.sql import Row
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, LongType
)

class AlphaVantageIngestor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.api_cfg = cfg["alpha_vantage"]

    def fetch(self):
        params = {
            "function": self.api_cfg["function"],
            "symbol": self.api_cfg["symbol"],
            "interval": self.api_cfg["interval"],
            "outputsize": self.api_cfg["output_size"],
            "apikey": self.api_cfg["api_key"],
        }

        response = requests.get(
            "https://www.alphavantage.co/query",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def to_rows(self, payload):
        key = f"Time Series ({self.api_cfg['interval']})"

        if key not in payload:
            raise ValueError(
                f"Expected key '{key}' not found in API response. "
                f"Keys: {payload.keys()}"
            )

        rows = []
        for ts, values in payload[key].items():
            rows.append(
                Row(
                    timestamp=ts,
                    open=float(values["1. open"]),
                    high=float(values["2. high"]),
                    low=float(values["3. low"]),
                    close=float(values["4. close"]),
                    volume=int(values["5. volume"]),
                    symbol=self.api_cfg["symbol"],
                )
            )
        return rows

    def write_parquet(self, spark, rows):
        schema = StructType([
            StructField("timestamp", StringType(), False),
            StructField("open", DoubleType(), True),
            StructField("high", DoubleType(), True),
            StructField("low", DoubleType(), True),
            StructField("close", DoubleType(), True),
            StructField("volume", LongType(), True),
            StructField("symbol", StringType(), True),
        ])

        df = spark.createDataFrame(rows, schema=schema)

        (
            df
            .write
            .mode("overwrite")
            .parquet(self.cfg["paths"]["raw"])
        )

    def run(self, spark):
        payload = self.fetch()
        rows = self.to_rows(payload)
        self.write_parquet(spark, rows)

import os
import yaml

def load_config(
    config_path="config/config.yaml",
    spark_path="config/spark.yaml",
):
    # load main config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # resolve env vars (Alpha Vantage key)
    key = cfg["alpha_vantage"].get("api_key")
    if isinstance(key, str) and key.startswith("${"):
        env_name = key.strip("${}")
        cfg["alpha_vantage"]["api_key"] = os.getenv(env_name)

    if not cfg["alpha_vantage"]["api_key"]:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set")

    # load spark config
    with open(spark_path, "r") as f:
        spark_cfg = yaml.safe_load(f)

    cfg["spark"] = spark_cfg["spark"]

    return cfg

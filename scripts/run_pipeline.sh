#!/bin/bash
set -e

source /home/ubuntu/projekt/venv/bin/activate
cd /home/ubuntu/projekt

python -m spark_jobs.ingest.alpha_vantage_ingest
python -m spark_jobs.preprocess.forex_daily_preprocess

python -m pipelines.sarima.train
python -m pipelines.sarima.backtest
python -m pipelines.naive.backtest
python -m pipelines.compare.select_model

python -m pipelines.sarima.predict
python -m pipelines.naive.predict
python -m pipelines.visualize.forecast_vs_actual

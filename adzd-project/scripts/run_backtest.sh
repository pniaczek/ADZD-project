#!/bin/bash
set -euo pipefail

# --- CONFIG ---
PROJECT_DIR="${PROJECT_DIR:-/home/ec2-user/ADZD-project}"
VENV_PATH="${VENV_PATH:-$PROJECT_DIR/.venv}"
SPARK_HOME="${SPARK_HOME:-/opt/spark}"
EVENT_LOG_DIR="${EVENT_LOG_DIR:-/opt/spark-events}"

# Backtest params (override by env vars)
FREQUENCY="${FREQUENCY:-daily}"
INGEST_DATE="${INGEST_DATE:-latest}"     # 'latest' or YYYY-MM-DD
SEASONAL_LAG="${SEASONAL_LAG:-5}"
WRITE_PREDS="${WRITE_PREDS:-0}"          # 1 -> write preds parquet

cd "$PROJECT_DIR"

# --- ENV ---
if [ -f "$VENV_PATH/bin/activate" ]; then
  source "$VENV_PATH/bin/activate"
fi

export PYTHONPATH="$PROJECT_DIR"
export PATH="$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH"

# --- DEPS ZIP (for Spark executors) ---
zip -r deps.zip config utils pipelines >/dev/null

echo "=== BACKTEST START $(date -u) ==="
echo "Project: $PROJECT_DIR"
echo "Frequency: $FREQUENCY"
echo "Ingest date: $INGEST_DATE"
echo "Event logs: $EVENT_LOG_DIR"
echo "Write preds: $WRITE_PREDS"
echo

# helper: adds --write-preds if WRITE_PREDS=1
WRITE_PREDS_ARG=""
if [ "$WRITE_PREDS" = "1" ]; then
  WRITE_PREDS_ARG="--write-preds"
fi

# --- NAIVE ---
spark-submit \
  --master local[*] \
  --name "forex-${FREQUENCY}-backtest-naive" \
  --conf spark.eventLog.enabled=true \
  --conf spark.eventLog.dir="$EVENT_LOG_DIR" \
  --py-files deps.zip \
  spark_jobs/backtest/forex_daily_backtest.py \
  --frequency "$FREQUENCY" \
  --ingest-date "$INGEST_DATE" \
  --model naive \
  $WRITE_PREDS_ARG

echo

# --- SEASONAL NAIVE ---
spark-submit \
  --master local[*] \
  --name "forex-${FREQUENCY}-backtest-seasonal_naive-lag${SEASONAL_LAG}" \
  --conf spark.eventLog.enabled=true \
  --conf spark.eventLog.dir="$EVENT_LOG_DIR" \
  --py-files deps.zip \
  spark_jobs/backtest/forex_daily_backtest.py \
  --frequency "$FREQUENCY" \
  --ingest-date "$INGEST_DATE" \
  --model seasonal_naive \
  --seasonal-lag "$SEASONAL_LAG" \
  $WRITE_PREDS_ARG

echo
echo "=== BACKTEST END $(date -u) ==="

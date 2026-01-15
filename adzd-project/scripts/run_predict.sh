#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ec2-user/ADZD-project}"
VENV_PATH="${VENV_PATH:-$PROJECT_DIR/.venv}"
SPARK_HOME="${SPARK_HOME:-/opt/spark}"
EVENT_LOG_DIR="${EVENT_LOG_DIR:-/opt/spark-events}"

FREQUENCY="${FREQUENCY:-daily}"
INGEST_DATE="${INGEST_DATE:-latest}"
SEASONAL_LAG="${SEASONAL_LAG:-5}"

cd "$PROJECT_DIR"

if [ -f "$VENV_PATH/bin/activate" ]; then
  source "$VENV_PATH/bin/activate"
fi

export PYTHONPATH="$PROJECT_DIR"
export PATH="$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH"

zip -r deps.zip config utils pipelines >/dev/null

echo "=== PREDICT START $(date -u) ==="
echo "Frequency: $FREQUENCY"
echo "Ingest date: $INGEST_DATE"
echo "Event logs: $EVENT_LOG_DIR"
echo

# --- NAIVE ---
spark-submit \
  --master local[*] \
  --name "forex-${FREQUENCY}-predict-naive" \
  --conf spark.eventLog.enabled=true \
  --conf spark.eventLog.dir="$EVENT_LOG_DIR" \
  --py-files deps.zip \
  spark_jobs/predict/forex_daily_predict.py \
  --frequency "$FREQUENCY" \
  --ingest-date "$INGEST_DATE" \
  --model naive

echo

# --- SEASONAL NAIVE ---
spark-submit \
  --master local[*] \
  --name "forex-${FREQUENCY}-predict-seasonal_naive-lag${SEASONAL_LAG}" \
  --conf spark.eventLog.enabled=true \
  --conf spark.eventLog.dir="$EVENT_LOG_DIR" \
  --py-files deps.zip \
  spark_jobs/predict/forex_daily_predict.py \
  --frequency "$FREQUENCY" \
  --ingest-date "$INGEST_DATE" \
  --model seasonal_naive \
  --seasonal-lag "$SEASONAL_LAG"

echo
echo "=== PREDICT END $(date -u) ==="

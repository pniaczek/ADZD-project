#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# --- venv ---
source .venv/bin/activate

# --- env ---
export PYTHONPATH="$(pwd)"
export SPARK_HOME=/opt/spark
export PATH="$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH"

RUN_TS="$(date -u +"%Y-%m-%dT%H%M%SZ")"
LOG_DIR="logs/pipeline"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_${RUN_TS}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== DAILY PIPELINE START ${RUN_TS} ==="
echo "PWD=$(pwd)"
python3 --version
spark-submit --version | head -n 2 || true

# rebuild deps.zip for spark-submit --py-files
zip -r deps.zip config utils pipelines >/dev/null

# -----------------------
# A) INGEST (Spark)
# -----------------------
spark-submit \
  --master local[*] \
  --name "alpha-vantage-fx-daily-ingest" \
  --conf spark.eventLog.enabled=true \
  --conf spark.eventLog.dir=/opt/spark-events \
  --py-files deps.zip \
  spark_jobs/ingest/alpha_vantage_ingest.py

# -----------------------
# B) PREPROCESS -> FEATURES (Spark)
# -----------------------
spark-submit \
  --master local[*] \
  --name "forex-daily-preprocess-features" \
  --conf spark.eventLog.enabled=true \
  --conf spark.eventLog.dir=/opt/spark-events \
  --py-files deps.zip \
  spark_jobs/preprocess/forex_daily_preprocess.py

# -----------------------
# C) BACKTEST (Python models)
# -----------------------
python3 -m pipelines.naive.backtest --pairs all
python3 -m pipelines.sarima.backtest --pairs all
python3 -m pipelines.prophet.backtest --pairs all


# -----------------------
# D) MODEL SELECTION (Python)
# (upewnij się, że to istnieje i obsługuje per pair)
# -----------------------
python3 -m pipelines.compare.select_model --pairs all || true

# -----------------------
# E) PREDICT (Python models)
# (upewnij się, że predict.py w naive/sarima/prophet jest już pod multi-pair)
# -----------------------
python3 -m pipelines.naive.predict --pairs all || true
python3 -m pipelines.sarima.predict --pairs all || true
python3 -m pipelines.prophet.predict --pairs all || true

# -----------------------
# VISUALIZE
# -----------------------
python3 -m pipelines.visualize.forecast_vs_actual --pairs all --models naive,sarima,prophet --no-intervals


# -----------------------
# OPTIONAL: MCP agent (jeśli masz agent.py i działa w tym repo)
# -----------------------
if [[ -f "agent.py" ]]; then
  mkdir -p logs/agent
  python3 agent.py > "logs/agent/agent_${RUN_TS}.txt" 2>&1 || true
  echo "Agent output: logs/agent/agent_${RUN_TS}.txt"
fi

echo "=== DAILY PIPELINE END ${RUN_TS} ==="

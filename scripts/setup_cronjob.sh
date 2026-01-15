#!/bin/bash
set -e

JOB='@hourly /home/ec2-user/ADZD-project/scripts/run_daily_pipeline.sh >> /home/ec2-user/script.log 2>&1'

# If the job is already present, do nothing
if crontab -l 2>/dev/null | grep -F -- "$JOB" >/dev/null; then
  echo "Cron job already installed"
  exit 0
fi

# Append the job
( crontab -l 2>/dev/null; echo "$JOB" ) | crontab -
echo "Cron job installed"

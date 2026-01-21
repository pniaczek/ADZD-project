#!/bin/bash
set -e

SPARK_HOME="/opt/spark"
JAVA_HOME="/usr/lib/jvm/java-17-amazon-corretto"
SERVICE_PATH="/etc/systemd/system/spark-history-server.service"
USER="ec2-user"
GROUP="ec2-user"


mkdir -p /opt/spark-events

sudo bash -c "cat > $SERVICE_PATH" <<EOF
[Unit]
Description=Apache Spark History Server
After=network.target

[Service]
Type=forking
User=$USER
Group=$GROUP
Environment="SPARK_HOME=$SPARK_HOME"
Environment="JAVA_HOME=$JAVA_HOME"
ExecStart=$SPARK_HOME/sbin/start-history-server.sh
ExecStop=$SPARK_HOME/sbin/stop-history-server.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd, enable and start service
sudo systemctl daemon-reload
sudo systemctl enable spark-history-server
sudo systemctl restart spark-history-server

echo "spark-history-server.service installed and started."

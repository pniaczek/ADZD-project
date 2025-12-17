### Commands for potential ownership issues
```
docker-compose exec -T spark-master id -u spark
docker-compose exec -T spark-master id -g spark
```
```
sudo groupadd -g <SPARK_GID> sparkcont || true
sudo usermod -aG sparkcont $USER
sudo chown -R $(id -u):<SPARK_GID> logs/spark-events
sudo chmod -R 2775 logs/spark-events
chmod +x scripts/run_pipeline.sh
```
restart docker containers
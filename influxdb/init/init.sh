#!/bin/bash
# InfluxDB 初始化脚本
# 此脚本在 InfluxDB 首次启动时自动运行

set -e

echo "开始初始化 InfluxDB..."

# 等待 InfluxDB 准备就绪
until curl -s http://localhost:8086/health > /dev/null 2>&1; do
    echo "等待 InfluxDB 启动..."
    sleep 2
done

echo "InfluxDB 已就绪"

# 设置 API token（如果需要额外创建）
influx auth list || true

echo "InfluxDB 初始化完成!"
echo "组织: jetson"
echo "Bucket: person_detection"
echo "Token: ${DOCKER_INFLUXDB_INIT_ADMIN_TOKEN}"

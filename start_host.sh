#!/bin/bash
# 在宿主机直接运行 YOLO 应用（解决 GMSL 摄像头在容器内无法工作的问题）
# InfluxDB 和 Grafana 仍然在 Docker 中运行

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== YOLO11 宿主机启动脚本 ==="

# 确保 InfluxDB 和 Grafana 容器在运行
echo "启动 InfluxDB 和 Grafana..."
docker compose up -d influxdb grafana

echo "等待 InfluxDB 就绪..."
until docker exec yolo11-influxdb curl -sf http://localhost:8086/health > /dev/null 2>&1; do
    sleep 2
done
echo "InfluxDB 就绪"

# 使用 ultralytics_v8 conda 环境运行 YOLO（后台守护进程）
echo "启动 YOLO 检测..."
nohup conda run -n ultralytics_v8 python3 yolo11n_grafana.py \
    --source 0 \
    --camera-type auto \
    --influx-url http://localhost:8086 \
    --influx-token XcOGuS__bo4NKPEk0zBYlOBIRrhMXlufMaaVLgmFMObXts_mCF-43kgUWhHGKtQfTEuPITWcB57eI32qlGy5TA== \
    --influx-org jetson \
    --influx-bucket person_detection \
    --device auto \
    --imgsz 640 \
    --stream-width 640 \
    --headless \
    --web-port 5001 > /tmp/yolo11_host.log 2>&1 &
disown $!
echo "YOLO PID: $!"

echo ""
echo "访问地址:"
echo "  视频流:   http://localhost:5001"
echo "  Grafana:  http://localhost:3001 (admin/admin)"

#!/bin/bash

# YOLO11 Docker 启动脚本
# 用于自动配置并启动所有服务

set -e

echo "=== YOLO11 Person Detection Docker 启动脚本 ==="

# 获取宿主机 IP
HOST_IP=$(hostname -I | awk '{print $1}')
echo "检测到宿主机 IP: $HOST_IP"

# 复制配置文件模板并替换 IP
echo "配置 Grafana 数据源..."

# 创建数据源配置
mkdir -p grafana/provisioning/datasources
cat > grafana/provisioning/datasources/datasources.yml << EOF
apiVersion: 1

datasources:
  - name: InfluxDBJetson
    type: influxdb
    access: proxy
    url: http://${HOST_IP}:8086
    isDefault: true
    jsonData:
      defaultBucket: person_detection
      organization: jetson
      version: Flux
    secureJsonData:
      token: XcOGuS__bo4NKPEk0zBYlOBIRrhMXlufMaaVLgmFMObXts_mCF-43kgUWhHGKtQfTEuPITWcB57eI32qlGy5TA==
EOF

# 更新看板配置中的 IP
if [ -f "grafana/dashboards/yolo11-realtime.json" ]; then
    echo "更新 Grafana 看板 IP..."
    sed -i "s/192\.168\.[0-9]*\.[0-9]*/${HOST_IP}/g" grafana/dashboards/yolo11-realtime.json
fi

# 启动 Docker Compose
echo "启动 Docker 容器..."
sudo docker-compose up -d

# 等待服务启动
echo "等待服务启动..."
sleep 10

# 检查状态
echo ""
echo "=== 服务状态 ==="
sudo docker-compose ps

echo ""
echo "=== 启动完成 ==="
echo "YOLO11 Web: http://localhost:5001"
echo "Grafana: http://localhost:3001 (admin/admin)"
echo "InfluxDB: http://localhost:8086"
echo ""
echo "Grafana 看板: http://localhost:3001/d/yolo11-realtime/yolo11n-e5ae9e-e697b6"

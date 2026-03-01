#!/bin/bash
# YOLO11 本地安装脚本（非 Docker 模式）

echo "=== YOLO11 本地安装脚本 ==="

# 1. 安装 InfluxDB
echo "[1/4] 安装 InfluxDB..."
if ! command -v influxd &> /dev/null; then
    # 下载并安装 InfluxDB 2.x
    wget -q https://dl.influxdata.com/influxdb/releases/influxdb2-2.7.12-linux-arm64.tar.gz
    tar xzf influxdb2-2.7.12-linux-arm64.tar.gz
    sudo cp -r influxdb2-2.7.12-linux-arm64/usr/* /usr/local/
    sudo cp -r influxdb2-2.7.12-linux-arm64/etc /etc/influxdbv2
    rm -rf influxdb2-2.7.12-linux-arm64*
    echo "InfluxDB 安装完成"
else
    echo "InfluxDB 已安装"
fi

# 2. 安装 Grafana
echo "[2/4] 安装 Grafana..."
if ! command -v grafana-server &> /dev/null; then
    wget -q https://dl.grafana.com/oss/release/grafana_12.4.0_arm64.deb
    sudo dpkg -i grafana_12.4.0_arm64.deb
    rm grafana_12.4.0_arm64.deb
    echo "Grafana 安装完成"
else
    echo "Grafana 已安装"
fi

# 3. 配置 InfluxDB
echo "[3/4] 配置 InfluxDB..."

# 创建 InfluxDB 配置目录
sudo mkdir -p /etc/influxdbv2

# 启动 InfluxDB（后台运行）
influxd --engine-path=/var/lib/influxdb2 &
INFLUX_PID=$!
echo "InfluxDB PID: $INFLUX_PID"

# 等待 InfluxDB 启动
sleep 5

# 设置 InfluxDB
influx auth list || influx setup \
  --username admin \
  --password admin123456 \
  --org jetson \
  --bucket person_detection \
  --token XcOGuS__bo4NKPEk0zBYlOBIRrhMXlufMaaVLgmFMObXts_mCF-43kgUWhHGKtQfTEuPITWcB57eI32qlGy5TA== \
  --force

echo "InfluxDB 配置完成"

# 4. 配置 Grafana
echo "[4/4] 配置 Grafana..."

# 启动 Grafana（后台运行）
sudo grafana-server &
sleep 3

echo ""
echo "=== 安装完成 ==="
echo "InfluxDB: http://localhost:8086"
echo "Grafana: http://localhost:3000 (admin/admin)"

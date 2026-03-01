# YOLO11n Person Detection with InfluxDB & Grafana

## 项目简介

这是一个基于 YOLO11n 的人员检测系统，支持实时视频流处理、Heatmap 热力图生成，并将检测数据存储到 InfluxDB，最终通过 Grafana 看板可视化展示。

## 系统架构

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   YOLO11 App    │────▶│    InfluxDB     │◀────│    Grafana      │
│  (Python/Flask) │     │ (时序数据库)     │     │ (可视化看板)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
       │                       │                       │
   端口 5001              端口 8086               端口 3001
```

## 快速开始

### 1. 环境要求

- Docker & Docker Compose
- **NVIDIA Jetson**（推荐，已在 Jetson Orin/AGX 上验证）
  - 需安装 NVIDIA Container Runtime（`nvidia-docker2` 或 `docker-compose` >= 2.x）
- 或 x86 机器（需修改 `docker-compose.yml`，见[非 Jetson 部署](#非-jetson-x86-部署)）

### 2. 准备 PyTorch wheel 文件（Jetson 必须）

Jetson 需从本地 wheel 安装 PyTorch（aarch64 + CUDA 版本），标准 PyPI 无法提供：

```bash
# 方式一：从 Jetson AI Lab PyPI 下载（需联网）
pip download torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126 \
  --dest wheel/ \
  --no-deps \
  -f https://pypi.jetson-ai-lab.dev/jp6/cu126

# 方式二：直接访问 NVIDIA 下载页面获取
# https://developer.nvidia.com/embedded/downloads
```

确认以下文件存在：
```
wheel/
├── torch-2.7.0-cp310-cp310-linux_aarch64.whl
├── torchvision-0.22.0-cp310-cp310-linux_aarch64.whl
└── torchaudio-2.7.0-cp310-cp310-linux_aarch64.whl
```

### 3. 启动服务

```bash
cd yolo11_grafana_influxdb

# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps
```

### 3. 访问服务

| 服务 | 地址 | 默认账号 |
|------|------|----------|
| YOLO11 Web | http://localhost:5001 | - |
| Grafana | http://localhost:3001 | admin/admin |
| InfluxDB | http://localhost:8086 | admin/admin123456 |

### 4. 查看 Grafana 看板

打开浏览器访问：http://localhost:3001/d/yolo11-realtime/yolo11n-realtime

## 服务配置

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `INFLUX_URL` | http://influxdb:8086 | InfluxDB 地址 |
| `INFLUX_TOKEN` | (见下方) | InfluxDB Token |
| `INFLUX_ORG` | jetson | 组织名称 |
| `INFLUX_BUCKET` | person_detection | 存储桶名称 |
| `VIDEO_SOURCE` | /app/videos/people-detection.mp4 | 视频源 |
| `DEVICE` | auto | 设备 (auto/cpu/cuda) |
| `WEB_PORT` | 5000 | Web 服务端口 |

### 端口映射

| 容器 | 内部端口 | 外部端口 |
|------|----------|----------|
| influxdb | 8086 | 8086 |
| grafana | 3000 | 3001 |
| yolo11-app | 5000 | 5001 |

## 命令说明

### 常用命令

```bash
# 启动所有服务
docker-compose up -d

# 启动并查看日志
docker-compose up -d
docker-compose logs -f

# 停止所有服务
docker-compose down

# 停止并删除数据卷
docker-compose down -v

# 重启单个服务
docker-compose restart yolo11-app

# 查看服务状态
docker-compose ps

# 查看服务日志
docker-compose logs influxdb
docker-compose logs grafana
docker-compose logs yolo11-app
```

### 调试命令

```bash
# 进入容器内部
docker exec -it yolo11-app bash
docker exec -it yolo11-influxdb bash
docker exec -it yolo11-grafana bash

# 查看 Python 应用日志
docker logs yolo11-app

# 测试 InfluxDB 连接
curl -s http://localhost:8086/health

# 测试 Grafana API
curl -s -u admin:admin http://localhost:3001/api/health
```

## 自定义配置

### 修改视频源

**使用摄像头（默认）：**
```bash
# docker-compose.yml 中
VIDEO_SOURCE=0        # 第一个 USB 摄像头
VIDEO_SOURCE=1        # 第二个 USB 摄像头
```

**使用视频文件：**
1. 将视频文件放入 `videos/` 目录（需自行创建）
2. 在 `docker-compose.yml` 中取消注释视频挂载：
   ```yaml
   - ./videos:/app/videos:ro
   ```
3. 修改 `VIDEO_SOURCE=/app/videos/your_video.mp4`
4. 注释掉 `devices:` 下的摄像头 passthrough
5. 重启：`docker-compose restart yolo11-app`

### 修改 InfluxDB Token

1. 修改 `docker-compose.yml` 中的 `DOCKER_INFLUXDB_INIT_ADMIN_TOKEN`
2. 修改 `grafana/provisioning/datasources/datasources.yml` 中的 token
3. 修改 `yolo11-app` 环境变量中的 `INFLUX_TOKEN`
4. 重新部署：`docker-compose down && docker-compose up -d`

### 修改 Grafana 看板

1. 访问 http://localhost:3001
2. 登录账号 admin/admin
3. 修改看板后，通过 API 导出：
```bash
curl -s -u admin:admin http://localhost:3001/api/dashboards/uid/yolo11-realtime | \
  python3 -c "import json,sys; d=json.load(sys.stdin); d['dashboard'].pop('id'); d['dashboard'].pop('version'); json.dump(d, open('grafana/dashboards/yolo11-realtime.json','w'), indent=2)"
```

## 数据持久化

- `influxdb-data`: InfluxDB 数据
- `grafana-data`: Grafana 数据和配置

删除数据卷命令：
```bash
docker-compose down -v
```

## 非 Jetson (x86) 部署

在标准 x86 机器上运行需修改 `docker-compose.yml`：

1. **移除 Jetson 专属挂载**（`/usr/local/cuda`、`/usr/lib/aarch64-linux-gnu`、`/usr/lib/python3.10/dist-packages` 三行 volumes）
2. **移除 Jetson 专属环境变量**（`NVIDIA_VISIBLE_DEVICES`、`NVIDIA_DRIVER_CAPABILITIES`、`LD_LIBRARY_PATH`、`PYTHONPATH`）
3. **修改 Dockerfile** 改为从 PyPI 安装 PyTorch：
   ```dockerfile
   # 替换本地 wheel 安装为网络安装
   RUN pip install --no-cache-dir torch torchvision torchaudio \
       --index-url https://download.pytorch.org/whl/cu118
   ```
4. 如使用摄像头，确认摄像头设备路径（`/dev/video0` 等）

## 故障排除

### 服务启动失败

```bash
# 查看详细日志
docker-compose logs

# 检查端口占用
netstat -tuln | grep -E '8086|3001|5001'
```

### InfluxDB 连接失败

```bash
# 检查 InfluxDB 是否启动
docker ps | grep influxdb

# 查看 InfluxDB 日志
docker logs yolo11-influxdb

# 测试连接
curl -s http://localhost:8086/health
```

### Grafana 看板不显示数据

1. 检查数据源配置是否正确
2. 检查 InfluxDB 是否有数据：
```bash
curl -s -X POST http://localhost:8086/api/v2/query?org=jetson \
  -H "Authorization: Token YOUR_TOKEN" \
  -H "Content-Type: application/vnd.flux" \
  -d 'from(bucket: "person_detection") |> range(start: -5m)'
```

### GPU 不可用

确保：
1. 主机已安装 NVIDIA 驱动
2. Docker 已配置 NVIDIA 运行时
3. `docker-compose.yml` 中已启用 GPU 支持

## 项目结构

```
yolo11_grafana_influxdb/
├── docker-compose.yml          # Docker Compose 配置
├── Dockerfile                  # YOLO11 应用镜像
├── .env.example                # 环境变量示例（cp .env.example .env）
├── requirements.txt            # Python 依赖（本地运行用）
├── yolo11n_grafana.py         # 主程序
├── yolo11n.pt                 # YOLO11n 模型权重（已入库）
├── wheel/                     # Jetson PyTorch wheel（需自行下载，见上方说明）
│   ├── torch-2.7.0-cp310-cp310-linux_aarch64.whl
│   ├── torchvision-0.22.0-cp310-cp310-linux_aarch64.whl
│   └── torchaudio-2.7.0-cp310-cp310-linux_aarch64.whl
├── videos/                    # 视频目录（gitignored，需自行创建和放置视频）
├── influxdb/
│   └── init/                  # InfluxDB 初始化脚本
├── grafana/
│   ├── provisioning/          # Grafana 自动配置
│   │   ├── datasources/      # 数据源配置
│   │   └── dashboards/       # 看板配置
│   └── dashboards/            # 看板 JSON 文件
└── README.md                  # 本文档
```

## 技术栈

- **YOLO11n**: Ultralytics 目标检测模型
- **InfluxDB**: 时序数据库
- **Grafana**: 数据可视化平台
- **Flask**: Python Web 框架
- **OpenCV**: 图像处理
- **PyTorch**: 深度学习框架

## 许可证

MIT License

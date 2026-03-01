# YOLO11 人员检测系统 - 本地运行完整教程

## 一、项目包含文件清单

本项目文件夹已包含所有需要的软件和文件，客户无需额外下载：

| 文件/目录 | 说明 | 是否必须 |
|-----------|------|----------|
| **wheel/** | PyTorch GPU 预编译包（torch-2.7.0、torchvision-0.22.0） | ✅ 必须 |
| **yolo11n.pt** | YOLO11n 模型权重文件（约 5MB） | ✅ 必须 |
| **videos/people-detection.mp4** | 测试用示例视频 | ✅ 必须 |
| **requirements.txt** | Python 依赖列表 | ✅ 必须 |
| **docker-compose.yml** | InfluxDB + Grafana 服务配置 | ✅ 必须 |
| **grafana/** | Grafana 看板和数据源配置 | ✅ 必须 |
| **influxdb/** | InfluxDB 初始化配置 | ✅ 必须 |
| **Dockerfile** | YOLO11 容器镜像（Docker 部署用） | 可选 |

---

## 二、硬件要求

| 硬件 | 要求 |
|------|------|
| 开发板 | NVIDIA Jetson (Nano/NX/Orin) 或 x86 Linux PC |
| 存储空间 | **至少 15GB 可用空间** |
| 内存 | 建议 4GB 以上 |
| GPU | NVIDIA GPU（可选，GPU 模式更快） |

---

## 三、软件环境要求

### 3.1 基础软件

| 软件 | 版本要求 | 说明 |
|------|----------|------|
| **操作系统** | Ubuntu 22.04 / JetPack 6.x | Jetson  |
| **Python** | **3.8 - 3.10** | 建议 3.10（项目 wheel 基于 3.10）|
| **Docker** | 20.10+ | 用于运行 InfluxDB 和 Grafana |
| **Docker Compose** | 2.0+ | 用于编排服务 |
| **NVIDIA 驱动** | 最新版 | JetPack 自带，PC 需要单独安装 |

### 3.2 检查 Python 版本

```bash
python3 --version
# 输出应该是 Python 3.8.x / 3.9.x / 3.10.x
```

> ⚠️ **注意**：项目提供的 PyTorch wheel 是针对 **Python 3.10** 编译的，请确保使用 Python 3.10。

---

## 四、运行方式选择

本项目支持 **两种运行方式**：
- **方式一：Docker 部署**（推荐新手，一键启动）
- **方式二：本地直接运行**（性能更好，可使用 GPU 加速）

---

# 方式一：Docker 部署（推荐）

## 4.1 安装 Docker（如果未安装）

```bash
# 更新软件包
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

# 添加 Docker 官方 GPG 密钥
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# 添加 Docker 仓库
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 安装 Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 启动 Docker
sudo systemctl start docker
sudo systemctl enable docker

# 将当前用户加入 docker 组
sudo usermod -aG docker $USER
# 重新登录后生效
```

## 4.2 安装 nvidia-docker（GPU 支持）

```bash
# 添加 nvidia-docker 仓库
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 安装 nvidia-docker2
sudo apt update
sudo apt install -y nvidia-docker2

# 配置 Docker 使用 nvidia 运行时
sudo tee /etc/docker/daemon.json <<'EOF'
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF

# 重启 Docker
sudo systemctl restart docker
```

## 4.3 启动服务

```bash
# 进入项目目录
cd /home/seeed/yolo11_grafana_influxdb

# 启动所有服务（InfluxDB + Grafana + YOLO11）
sudo docker compose up -d
```

## 4.4 检查服务状态

```bash
sudo docker ps
```

应该看到 3 个容器运行：
- `yolo11-influxdb` - 端口 8086
- `yolo11-grafana` - 端口 3001  
- `yolo11-app` - 端口 5001

---

# 方式二：本地直接运行（推荐，性能更好）

## 5.1 安装 Python 依赖

```bash
# 进入项目目录
cd /home/seeed/yolo11_grafana_influxdb

# 安装 Python 依赖
pip3 install -r requirements.txt
```

## 5.2 安装 PyTorch GPU 版本（关键步骤！）

Jetson 设备**必须**安装 NVIDIA 官方编译的 PyTorch，不能用标准 pip 安装！

### 使用项目提供的 wheel 文件：

```bash
# 安装 PyTorch GPU 版本
pip3 install /home/seeed/yolo11_grafana_influxdb/wheel/torch-2.7.0-cp310-cp310-linux_aarch64.whl \
    /home/seeed/yolo11_grafana_influxdb/wheel/torchvision-0.22.0-cp310-cp310-linux_aarch64.whl \
    --no-deps
```

### 验证 CUDA 可用：

```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
```

输出应该显示：
```
CUDA available: True
Device count: 1
```

> ⚠️ **重要**：如果是 `CUDA available: False`，说明 PyTorch 安装的是 CPU 版本！

## 5.3 启动 InfluxDB 和 Grafana

```bash
# 进入项目目录
cd /home/seeed/yolo11_grafana_influxdb

# 启动 InfluxDB 和 Grafana（Docker 方式）
sudo docker compose up -d influxdb grafana

# 等待服务启动
sleep 10

# 验证 InfluxDB
curl -s http://localhost:8086/health
```

## 5.4 运行 YOLO11 应用

### GPU 模式（推荐）

```bash
cd /home/seeed/yolo11_grafana_influxdb

INFLUX_URL=http://localhost:8086 \
INFLUX_TOKEN=XcOGuS__bo4NKPEk0zBYlOBIRrhMXlufMaaVLgmFMObXts_mCF-43kgUWhHGKtQfTEuPITWcB57eI32qlGy5TA== \
INFLUX_ORG=jetson \
INFLUX_BUCKET=person_detection \
VIDEO_SOURCE=/home/seeed/yolo11_grafana_influxdb/videos/people-detection.mp4 \
DEVICE=0 \
WEB_PORT=5001 \
python3 yolo11n_grafana.py --headless
```

---

## 六、访问服务

服务启动后，通过以下地址访问：

| 服务 | 地址 | 默认账号 | 说明 |
|------|------|----------|------|
| **YOLO11 检测界面** | http://jetson-ip:5001 | - | 实时视频流 + 热力图 |
| **Grafana 看板** | http://jetson-ip:3001 | admin / admin | 数据可视化 |
| **InfluxDB** | http://jetson-ip:8086 | admin / admin123456 | 时序数据库 |

> 📌 获取 IP：`hostname -I`

### 查看 Grafana 看板

1. 打开浏览器访问：`http://<your-ip>:3001`
2. 登录：账号 `admin`，密码 `admin`
3. 进入：`Dashboards` → `yolo11-realtime`

---

## 七、使用说明

### 7.1 修改视频源

**使用本地视频：**
```bash
# 复制视频到 videos 目录
cp /path/to/video.mp4 ./videos/

# 运行
VIDEO_SOURCE=/home/seeed/yolo11_grafana_influxdb/videos/video.mp4 python3 yolo11n_grafana.py --headless
```

**使用 USB 摄像头：**
```bash
# 检查摄像头
ls -la /dev/video*

# 使用摄像头 0
VIDEO_SOURCE=0 python3 yolo11n_grafana.py --headless
```

### 7.2 常用命令

```bash
# Docker 方式启动
sudo docker compose up -d

# Docker 方式停止
sudo docker compose down

# 本地运行
python3 yolo11n_grafana.py --headless

# 查看调试信息
curl http://localhost:5001/debug.json
```

---

## 八、故障排查

### 8.1 服务启动失败

```bash
# 查看日志
sudo docker compose logs

# 检查端口
sudo netstat -tuln | grep -E '8086|3001|5001'

# 检查容器
sudo docker ps -a
```

### 8.2 InfluxDB 连接失败

```bash
# 检查容器
sudo docker ps | grep influxdb

# 测试连接
curl -s http://localhost:8086/health
```

### 8.3 GPU 不可用

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# 检查已安装的 PyTorch
pip3 list | grep torch
```

### 8.4 磁盘空间不足

```bash
# 查看磁盘
df -h

# 清理 Docker
sudo docker system prune -af --volumes
```

---

## 九、环境变量参考

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `INFLUX_URL` | http://localhost:8086 | InfluxDB 地址 |
| `INFLUX_TOKEN` | XcOGuS__bo4NKPEk... | InfluxDB Token |
| `INFLUX_ORG` | jetson | 组织名称 |
| `INFLUX_BUCKET` | person_detection | 存储桶名称 |
| `VIDEO_SOURCE` | videos/people-detection.mp4 | 视频源 |
| `DEVICE` | 0 | 设备 (0=GPU, cpu=CPU) |
| `WEB_PORT` | 5001 | Web 服务端口 |

---

## 十、快速验证清单

- [ ] Python 3.10 已安装
- [ ] Docker 和 Docker Compose 已安装
- [ ] PyTorch GPU 版本已安装（`torch.cuda.is_available() = True`）
- [ ] 项目文件完整（wheel/、yolo11n.pt、videos/ 等）
- [ ] InfluxDB 和 Grafana 已启动
- [ ] YOLO11 应用已运行
- [ ] http://localhost:5001 可访问（视频流）
- [ ] http://localhost:3001 可访问（Grafana）

---

**祝使用愉快！** 🎉

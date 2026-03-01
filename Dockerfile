# YOLO11n Person Detection with InfluxDB & Grafana
# 基于本地已有的 nvidia/cuda:11.8.0-base-ubuntu22.04（Python 3.10，无需联网拉镜像）
# runtime: nvidia 运行时自动挂载宿主机 CUDA 驱动，实现 GPU 推理
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libopenblas-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# 从本地 wheel 安装 PyTorch（CUDA 版，不联网）
COPY wheel/torch-2.7.0-cp310-cp310-linux_aarch64.whl /tmp/
COPY wheel/torchvision-0.22.0-cp310-cp310-linux_aarch64.whl /tmp/
COPY wheel/torchaudio-2.7.0-cp310-cp310-linux_aarch64.whl /tmp/

RUN pip install --no-cache-dir \
    /tmp/torch-2.7.0-cp310-cp310-linux_aarch64.whl \
    /tmp/torchvision-0.22.0-cp310-cp310-linux_aarch64.whl \
    /tmp/torchaudio-2.7.0-cp310-cp310-linux_aarch64.whl \
    && rm /tmp/*.whl

# 安装其他依赖
RUN pip install --no-cache-dir \
    ultralytics \
    influxdb-client \
    flask \
    werkzeug \
    "numpy<2"

# 拷贝应用文件
COPY yolo11n_grafana.py /app/
COPY yolo11n.pt /app/

RUN mkdir -p /app/videos

ENV PYTHONUNBUFFERED=1

EXPOSE 5000

ENTRYPOINT ["python", "yolo11n_grafana.py"]
CMD ["--headless"]

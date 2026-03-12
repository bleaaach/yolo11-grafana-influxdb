# YOLO11n Person Detection with InfluxDB & Grafana
# 基于本地已有的 nvidia/cuda:11.8.0-base-ubuntu22.04（Python 3.10，无需联网拉镜像）
# runtime: nvidia 运行时自动挂载宿主机 CUDA 驱动，实现 GPU 推理
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖（含 GStreamer，用于 USB 摄像头低延迟采集）
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libopenblas-dev \
    curl \
    python3-opencv \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-tools \
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

# 安装其他依赖（含 onnx，用于首次启动导出 TensorRT engine）
# opencv 用系统包（python3-opencv），自带 GStreamer 支持，不用 pip 版本
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ \
    "ultralytics[export]" \
    influxdb-client \
    flask \
    werkzeug \
    "numpy<2" \
    "onnx>=1.12.0,<2.0.0" \
    onnxslim \
    && pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true

# 拷贝应用文件
COPY yolo11n_grafana.py /app/
COPY yolo11n.pt /app/
COPY entrypoint.sh /app/

RUN mkdir -p /app/videos /app/models \
    && chmod +x /app/entrypoint.sh

ENV PYTHONUNBUFFERED=1

EXPOSE 5000

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--headless"]

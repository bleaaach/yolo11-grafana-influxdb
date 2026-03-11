#!/bin/bash
set -e

ENGINE_DIR="/app/models"
ENGINE_PATH="${ENGINE_DIR}/yolo11n.engine"
MODEL_PATH="/app/yolo11n.pt"
IMGSZ="${IMGSZ:-640}"

mkdir -p "$ENGINE_DIR"

# 首次启动时自动生成 TensorRT engine（耗时约 2-5 分钟）
if [ ! -f "$ENGINE_PATH" ]; then
    echo "=========================================="
    echo " TensorRT engine 不存在，开始自动生成..."
    echo " 模型: $MODEL_PATH"
    echo " imgsz: $IMGSZ"
    echo " 此过程仅在首次启动时执行，约需 2-5 分钟"
    echo "=========================================="
    python -c "
from ultralytics import YOLO
model = YOLO('${MODEL_PATH}')
model.export(format='engine', device=0, imgsz=${IMGSZ}, half=True)
"
    # ultralytics 导出的 engine 文件在 .pt 同目录下
    GENERATED="${MODEL_PATH%.pt}.engine"
    if [ -f "$GENERATED" ]; then
        mv "$GENERATED" "$ENGINE_PATH"
        echo "=========================================="
        echo " TensorRT engine 生成完成！"
        echo " 已保存到: $ENGINE_PATH"
        echo "=========================================="
    else
        echo "[WARN] engine 文件未找到，将使用 PyTorch 模式运行"
    fi
else
    echo "[INFO] 已检测到 TensorRT engine: $ENGINE_PATH"
fi

# 如果 engine 存在，通过环境变量传递给应用
if [ -f "$ENGINE_PATH" ]; then
    export YOLO_MODEL="$ENGINE_PATH"
else
    export YOLO_MODEL="$MODEL_PATH"
fi

# 启动主应用
exec python yolo11n_grafana.py "$@"

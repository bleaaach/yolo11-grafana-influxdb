import cv2
import time
import argparse
import numpy as np
from ultralytics import YOLO
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import WriteOptions
import os
import sys
import threading
import gc
import queue
from flask import Flask, Response, jsonify
import json

# Flask app for streaming
app = Flask(__name__)
output_frame = None
frame_lock = threading.Lock()
state_lock = threading.Lock()
runtime_state = {
    "ts": 0.0,
    "frame_idx": 0,
    "source": "",
    "person_count": 0,
    "total_visitors": 0,
    "avg_confidence": 0.0,
    "confidences": [],
    "centers": [],
    "influx": {
        "last_person_write_ts": 0.0,
        "bucket": "",
        "org": "",
        "write_latency_ms": 0.0
    },
    "performance": {
        "fps": 0.0,
        "display_fps": 0.0,
        "inference_ms": 0.0
    }
}

def generate():
    global output_frame, frame_lock
    last_frame = None
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75]  # 降低质量减少文件大小和编码时间
    while True:
        with frame_lock:
            frame = output_frame  # 取引用，不 copy
        if frame is None:
            time.sleep(0.02)
            continue
        # 跳过与上次相同的帧（主循环 33ms 更新一次，generate 可能更快）
        if frame is last_frame:
            time.sleep(0.005)
            continue
        last_frame = frame
        frame_copy = frame.copy()  # 拿到引用后再 copy，避免长时间持锁
        (flag, encodedImage) = cv2.imencode(".jpg", frame_copy, encode_params)
        if not flag:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Cache-Control: no-store\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    resp = Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    resp.headers["X-Accel-Buffering"] = "no"  # 禁用 nginx 缓冲（如有反向代理）
    return resp

@app.route("/")
def index():
    return "<h1>YOLO11n Heatmap Stream</h1><img src='/video_feed'>"

@app.route("/debug.json")
def debug_json():
    with state_lock:
        return jsonify(runtime_state)

@app.route("/debug")
def debug_page():
    with state_lock:
        s = json.dumps(runtime_state, ensure_ascii=False, indent=2)
    return "<html><head><meta charset='utf-8'><meta http-equiv='refresh' content='1'><title>Debug</title></head><body><h2>Runtime State</h2><pre>"+s+"</pre></body></html>"

class InfluxDBSender:
    def __init__(self, url, token, org, bucket):
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.client = None
        self.write_api = None
        self._queue = queue.Queue(maxsize=20)
        self._worker = None
        self.connect()

    def connect(self):
        try:
            self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
            from influxdb_client.client.write_api import SYNCHRONOUS
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            # 启动异步写入线程，避免网络 I/O 阻塞主循环
            self._worker = threading.Thread(target=self._write_worker, daemon=True)
            self._worker.start()
            print(f"成功连接到 InfluxDB: {self.url}")
        except Exception as e:
            print(f"连接 InfluxDB 失败: {e}")
            self.client = None

    def _write_worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            points = item
            try:
                self.write_api.write(bucket=self.bucket, record=points)
            except Exception as e:
                print(f"写入失败: {e}")
            self._queue.task_done()

    def send_person_count(self, count, timestamp=None):
        if self.write_api is None:
            return False
        point = Point("person_count") \
            .tag("source", "yolo11n_camera") \
            .tag("device", "jetson") \
            .field("count", int(count))
        if timestamp:
            point = point.time(timestamp)
        try:
            self._queue.put_nowait(point)
            return True
        except queue.Full:
            return False

    def send_all(self, count, avg_confidence, total_visitors, timestamp=None):
        """合并写入 person_count 和 detection_details，减少网络往返"""
        if self.write_api is None:
            return False
        p1 = Point("person_count") \
            .tag("source", "yolo11n_camera") \
            .tag("device", "jetson") \
            .field("count", int(count)) \
            .field("total_visitors", int(total_visitors))
        p2 = Point("detection_details") \
            .tag("source", "yolo11n_camera") \
            .tag("device", "jetson") \
            .field("person_count", int(count)) \
            .field("avg_confidence", float(avg_confidence))
        if timestamp:
            p1 = p1.time(timestamp)
            p2 = p2.time(timestamp)
        try:
            self._queue.put_nowait([p1, p2])
            return True
        except queue.Full:
            return False

    def send_detection_details(self, count, avg_confidence, timestamp=None):
        if self.write_api is None:
            return False
        point = Point("detection_details") \
            .tag("source", "yolo11n_camera") \
            .tag("device", "jetson") \
            .field("person_count", int(count)) \
            .field("avg_confidence", float(avg_confidence))
        if timestamp:
            point = point.time(timestamp)
        try:
            self._queue.put_nowait(point)
            return True
        except queue.Full:
            return False

    def send_uptime(self, uptime_seconds, start_time=None):
        if self.write_api is None:
            return False
        point = Point("app_status") \
            .tag("source", "yolo11n_camera") \
            .tag("device", "jetson") \
            .field("uptime_seconds", float(uptime_seconds))
        if start_time:
            point = point.field("start_time", float(start_time))
        try:
            self._queue.put_nowait(point)
            return True
        except queue.Full:
            return False

    def close(self):
        if self._worker and self._worker.is_alive():
            self._queue.put(None)
            self._worker.join(timeout=3)
        if self.client:
            self.client.close()

class HeatmapGenerator:
    def __init__(self, width, height, alpha=0.5, decay=0.95, ksize=51, scale=0.5):
        self.width = width
        self.height = height
        self.alpha = alpha
        self.decay = decay
        # 热力图以缩小分辨率维护，降低计算量
        self.scale = scale
        self.hw = max(1, int(width * scale))
        self.hh = max(1, int(height * scale))
        self.heatmap = np.zeros((self.hh, self.hw), dtype=np.float32)
        if ksize % 2 == 0:
            ksize += 1
        # 根据缩放比例调整 kernel 大小，保持视觉效果一致
        ksize_scaled = max(3, int(ksize * scale))
        if ksize_scaled % 2 == 0:
            ksize_scaled += 1
        g = cv2.getGaussianKernel(ksize_scaled, -1)
        kernel = g @ g.T
        kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-6)
        self.kernel = kernel.astype(np.float32)
        self.half = ksize_scaled // 2

    def update(self, centers):
        self.heatmap *= self.decay
        h = self.half
        k = self.kernel
        for x, y in centers:
            sx = int(x * self.scale)
            sy = int(y * self.scale)
            if 0 <= sx < self.hw and 0 <= sy < self.hh:
                x0 = max(0, sx - h)
                y0 = max(0, sy - h)
                x1 = min(self.hw, sx + h + 1)
                y1 = min(self.hh, sy + h + 1)
                kx0 = h - (sx - x0)
                ky0 = h - (sy - y0)
                kx1 = kx0 + (x1 - x0)
                ky1 = ky0 + (y1 - y0)
                self.heatmap[y0:y1, x0:x1] += k[ky0:ky1, kx0:kx1]
        np.clip(self.heatmap, 0, 1.0, out=self.heatmap)

    def apply_to_frame(self, frame):
        # 在缩放分辨率下生成彩色热力图，再放大叠加
        heatmap_norm = (self.heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        heatmap_full = cv2.resize(heatmap_color, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return cv2.addWeighted(frame, 1 - self.alpha, heatmap_full, self.alpha, 0)

class Camera:
    """支持 USB 摄像头、GMSL/CSI 摄像头（通过 GStreamer）和视频文件"""

    # 预定义 GMSL GStreamer pipeline 模板
    GMSL_PIPELINE = (
        "v4l2src device=/dev/video{idx} ! "
        "video/x-raw,format=YUYV,width={w},height={h},framerate={fps}/1 ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
    )
    ARGUS_PIPELINE = (
        "nvarguscamerasrc sensor-id={idx} ! "
        "video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 ! "
        "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
        "video/x-raw,format=BGR ! appsink drop=1"
    )

    def __init__(self, source=0, width=1920, height=1080, fps=30, camera_type="auto"):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_type = camera_type  # auto / usb / gmsl / argus / gst / file
        self.cap = None
        # 后台读取线程相关
        self._bg_ret = False
        self._bg_frame = None
        self._bg_lock = threading.Lock()
        self._bg_running = False
        self._bg_thread = None

    def _build_gstreamer_pipeline(self, idx):
        """根据 camera_type 构造 GStreamer pipeline"""
        if self.camera_type == "argus":
            return self.ARGUS_PIPELINE.format(idx=idx, w=self.width, h=self.height, fps=self.fps)
        else:  # gmsl
            return self.GMSL_PIPELINE.format(idx=idx, w=self.width, h=self.height, fps=self.fps)

    def open(self):
        # 检查 source 是否为数字
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)

        # 如果 source 是完整 GStreamer pipeline 字符串（包含 ! ）
        if isinstance(self.source, str) and "!" in self.source:
            self.camera_type = "gst"
            print(f"使用自定义 GStreamer pipeline: {self.source}")
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                print(f"GStreamer pipeline 打开失败: {self.source}")
                return False
            return True

        # 视频文件
        if isinstance(self.source, str):
            self.camera_type = "file"
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                print(f"无法打开视频文件: {self.source}")
                return False
            return True

        # 数字索引 — 按 camera_type 选择打开方式
        idx = self.source
        if self.camera_type in ("gmsl", "argus"):
            pipeline = self._build_gstreamer_pipeline(idx)
            print(f"使用 {self.camera_type.upper()} GStreamer pipeline: {pipeline}")
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                print(f"GStreamer 打开失败，回退到 V4L2")
                self.cap = cv2.VideoCapture(idx)
        elif self.camera_type == "auto":
            # 自动检测：优先尝试 V4L2，失败则尝试 GMSL GStreamer
            self.cap = cv2.VideoCapture(idx)
            if not self.cap.isOpened():
                print(f"V4L2 /dev/video{idx} 打开失败，尝试 GMSL GStreamer...")
                pipeline = self._build_gstreamer_pipeline(idx)
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if self.cap.isOpened():
                    self.camera_type = "gmsl"
                    print(f"GMSL GStreamer 打开成功")
        else:
            # usb 或其他
            self.cap = cv2.VideoCapture(idx)

        if not self.cap.isOpened():
            print(f"无法打开源: {self.source} (类型: {self.camera_type})")
            return False

        # USB 摄像头设置参数
        if self.camera_type in ("auto", "usb"):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 缓冲区只保留1帧
        # 启动后台读取线程（仅摄像头模式，不用于视频文件）
        if self.camera_type != "file":
            self._start_bg_reader()
        return True

    def _start_bg_reader(self):
        """后台持续读取摄像头帧，主线程调用 read() 时直接取最新帧，不阻塞"""
        self._bg_running = True
        self._bg_thread = threading.Thread(target=self._bg_read_loop, daemon=True)
        self._bg_thread.start()

    def _bg_read_loop(self):
        while self._bg_running:
            ret, frame = self.cap.read()
            with self._bg_lock:
                self._bg_ret = ret
                self._bg_frame = frame

    def read(self):
        if self.cap is None:
            return False, None
        # 摄像头模式：从后台缓冲取最新帧
        if self._bg_running:
            with self._bg_lock:
                ret = self._bg_ret
                frame = self._bg_frame
            if frame is not None:
                return ret, frame.copy()
            # 后台线程还没读到第一帧，回退到直接读
            return self.cap.read()
        # 视频文件模式：直接读
        return self.cap.read()

    def release(self):
        self._bg_running = False
        if self._bg_thread:
            self._bg_thread.join(timeout=1)
        if self.cap:
            self.cap.release()

# 向后兼容别名
USBCamera = Camera

def inference_worker(model, camera, device, half, imgsz, result_dict, result_lock, running_flag):
    """独立推理线程：持续运行 YOLO，结果写入 result_dict，不阻塞显示循环"""
    frame_count = 0
    last_fps_time = time.time()
    last_gc_time = time.time()
    while running_flag[0]:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.01)
            continue
        t0 = time.time()
        results = model.predict(source=frame, classes=[0], verbose=False,
                                device=device, half=half, imgsz=imgsz)
        dt = time.time() - t0
        count = 0
        confidences = []
        centers = []
        for r in results:
            for box in r.boxes:
                count += 1
                confidences.append(float(box.conf[0]))
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
        del results
        with result_lock:
            result_dict["count"] = count
            result_dict["confidences"] = confidences
            result_dict["centers"] = centers
            result_dict["inference_ms"] = dt * 1000
        frame_count += 1
        now = time.time()
        elapsed = now - last_fps_time
        if elapsed >= 5.0:
            infer_fps = frame_count / elapsed
            frame_count = 0
            last_fps_time = now
            print(f"[推理] FPS: {infer_fps:.1f}, 推理耗时: {dt*1000:.1f}ms, 人数: {count}")
            with state_lock:
                runtime_state["performance"]["fps"] = round(infer_fps, 1)
                runtime_state["performance"]["inference_ms"] = round(dt * 1000, 1)
        if now - last_gc_time >= 60:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            last_gc_time = now


def main():
    parser = argparse.ArgumentParser(description='YOLO11n Person Detection with InfluxDB & Grafana')
    parser.add_argument('--source', type=str, default=os.getenv("VIDEO_SOURCE", "0"), help='Video source: 0 for webcam, or path to video file')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--influx-url', type=str, default=os.getenv("INFLUX_URL", "http://localhost:8086"), help='InfluxDB URL')
    parser.add_argument('--influx-token', type=str, default=os.getenv("INFLUX_TOKEN", "my-super-secret-auth-token"), help='InfluxDB Token')
    parser.add_argument('--influx-org', type=str, default=os.getenv("INFLUX_ORG", "jetson"), help='InfluxDB Org')
    parser.add_argument('--influx-bucket', type=str, default=os.getenv("INFLUX_BUCKET", "person_detection"), help='InfluxDB Bucket')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no display)')
    parser.add_argument('--blur', action='store_true', help='Blur detected persons')
    parser.add_argument('--heatmap-alpha', type=float, default=0.5, help='Heatmap transparency (0-1)')
    parser.add_argument('--device', type=str, default=os.getenv("DEVICE", "auto"), help='Device: auto/cpu/cuda or GPU index (e.g., 0)')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 on GPU')
    parser.add_argument('--heatmap-ksize', type=int, default=51, help='Heatmap Gaussian kernel size (odd)')
    parser.add_argument('--web-port', type=int, default=int(os.getenv("WEB_PORT", "5001")), help='Flask web server port')
    parser.add_argument('--camera-type', type=str, default=os.getenv("CAMERA_TYPE", "auto"),
                        choices=["auto", "usb", "gmsl", "argus", "gst"],
                        help='Camera type: auto/usb/gmsl/argus/gst (default: auto)')
    parser.add_argument('--imgsz', type=int, default=int(os.getenv("IMGSZ", "480")),
                        help='YOLO input size (smaller=faster, default: 480)')
    parser.add_argument('--stream-width', type=int, default=int(os.getenv("STREAM_WIDTH", "640")),
                        help='Display/stream width in pixels (default: 640); height derived from aspect ratio')

    args = parser.parse_args()

    # 初始化 InfluxDB
    influx = InfluxDBSender(args.influx_url, args.influx_token, args.influx_org, args.influx_bucket)

    # 初始化 YOLO 模型，优先使用本地 TensorRT engine
    script_dir = os.path.dirname(os.path.abspath(__file__))
    engine_path = os.path.join(script_dir, 'yolo11n.engine')
    pt_path = os.path.join(script_dir, 'yolo11n.pt')
    if os.path.exists('/app/yolo11n.engine'):
        model_path = '/app/yolo11n.engine'
    elif os.path.exists(engine_path):
        model_path = engine_path
    elif os.path.exists('/app/yolo11n.pt'):
        model_path = '/app/yolo11n.pt'
    elif os.path.exists(pt_path):
        model_path = pt_path
    else:
        model_path = 'yolo11n.pt'
    print(f"正在加载 YOLO11n 模型: {model_path}")
    model = YOLO(model_path)
    selected_device = args.device
    if selected_device == "auto":
        if os.path.exists('/dev/nvidia0'):
            selected_device = 0
        else:
            selected_device = "cpu"
    elif selected_device.isdigit():
        selected_device = int(selected_device)
    use_half = args.fp16 or (selected_device != "cpu")
    print(f"使用设备: {selected_device}, FP16: {use_half}, 输入尺寸: {args.imgsz}")

    # 启动 Flask 视频流线程
    t = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=args.web_port, debug=False, use_reloader=False, threaded=True))
    t.daemon = True
    t.start()

    # 初始化摄像头/视频
    camera = Camera(args.source, camera_type=args.camera_type)
    if not camera.open():
        sys.exit(1)

    cam_w = int(camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头分辨率: {cam_w}x{cam_h}")

    # 流/显示分辨率（远小于摄像头原始分辨率，降低热力图和编码开销）
    STREAM_W = args.stream_width
    STREAM_H = int(STREAM_W * cam_h / cam_w) if cam_w > 0 else int(STREAM_W * 9 / 16)
    scale_x = STREAM_W / cam_w if cam_w > 0 else 1.0
    scale_y = STREAM_H / cam_h if cam_h > 0 else 1.0
    print(f"流分辨率: {STREAM_W}x{STREAM_H}")

    # 热力图在流分辨率下运行（scale=1.0，已经足够小）
    heatmap_gen = HeatmapGenerator(STREAM_W, STREAM_H, alpha=args.heatmap_alpha,
                                   ksize=args.heatmap_ksize, scale=1.0)

    # 启动独立推理线程
    infer_result = {"count": 0, "confidences": [], "centers": [], "inference_ms": 0.0}
    infer_lock = threading.Lock()
    running_flag = [True]
    infer_thread = threading.Thread(
        target=inference_worker,
        args=(model, camera, selected_device, use_half, args.imgsz,
              infer_result, infer_lock, running_flag),
        daemon=True
    )
    infer_thread.start()
    print("推理线程已启动，显示循环目标 30fps")

    # 今日累计人流
    total_visitors = 0
    prev_person_count = 0
    last_day = time.localtime().tm_mday
    last_send_time = time.time()
    last_status_send = last_send_time
    start_time = last_send_time

    # 显示循环帧率统计
    disp_count = 0
    last_disp_fps_time = time.time()
    TARGET_DT = 1.0 / 30.0  # 目标 30fps

    try:
        while True:
            t_loop = time.time()

            # 从后台线程取最新摄像头帧（非阻塞）
            ret, frame = camera.read()
            if not ret:
                time.sleep(0.01)
                continue

            # 缩放到流分辨率（大帧 → 小帧，后续所有操作都在小帧上）
            small = cv2.resize(frame, (STREAM_W, STREAM_H), interpolation=cv2.INTER_LINEAR)
            del frame  # 立即释放大帧内存

            # 取最新推理结果（非阻塞，不等待推理完成）
            with infer_lock:
                person_count = infer_result["count"]
                confidences = list(infer_result["confidences"])
                raw_centers = list(infer_result["centers"])
                inference_ms = infer_result["inference_ms"]

            # 将原图坐标映射到流分辨率
            centers = [(int(x * scale_x), int(y * scale_y)) for x, y in raw_centers]

            # 模糊处理（在小帧上）
            if args.blur:
                for (cx, cy) in centers:
                    bx1 = max(0, cx - STREAM_W // 20)
                    by1 = max(0, cy - STREAM_H // 15)
                    bx2 = min(STREAM_W, cx + STREAM_W // 20)
                    by2 = min(STREAM_H, cy + STREAM_H // 15)
                    roi = small[by1:by2, bx1:bx2]
                    if roi.size > 0:
                        small[by1:by2, bx1:bx2] = cv2.GaussianBlur(roi, (31, 31), 10)
                    cv2.rectangle(small, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

            # 更新并叠加热力图（全在流分辨率下）
            heatmap_gen.update(centers)
            disp = heatmap_gen.apply_to_frame(small)

            # 文字叠加
            cv2.putText(disp, f"Person Count: {person_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 推送到 Flask 流（直接赋值，不 copy，节省内存带宽）
            global output_frame
            with frame_lock:
                output_frame = disp

            # 显示画面
            if not args.headless:
                try:
                    cv2.imshow('YOLO11n Person Detection', disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception:
                    pass

            # 定时写入 InfluxDB（每 200ms）
            current_time = time.time()
            if current_time - last_send_time >= 0.2:
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                write_timestamp = int(current_time * 1e9)
                current_day = time.localtime().tm_mday
                if current_day != last_day:
                    total_visitors = 0
                    last_day = current_day
                if person_count > prev_person_count:
                    total_visitors += (person_count - prev_person_count)
                prev_person_count = person_count
                influx.send_all(person_count, avg_conf, total_visitors, timestamp=write_timestamp)
                last_send_time = current_time
                with state_lock:
                    runtime_state["ts"] = current_time
                    runtime_state["frame_idx"] = -1
                    runtime_state["source"] = str(args.source)
                    runtime_state["person_count"] = int(person_count)
                    runtime_state["total_visitors"] = int(total_visitors)
                    runtime_state["avg_confidence"] = float(avg_conf)
                    runtime_state["confidences"] = [float(c) for c in confidences]
                    runtime_state["centers"] = [(int(x), int(y)) for (x, y) in raw_centers]
                    runtime_state["influx"]["last_person_write_ts"] = float(last_send_time)
                    runtime_state["influx"]["bucket"] = args.influx_bucket
                    runtime_state["influx"]["org"] = args.influx_org
                    runtime_state["influx"]["write_latency_ms"] = 0.0
                    runtime_state["performance"]["inference_ms"] = round(inference_ms, 1)

            # 定时发送运行状态（每 10 秒）
            if current_time - last_status_send >= 10.0:
                influx.send_uptime(uptime_seconds=current_time - start_time, start_time=start_time)
                last_status_send = current_time

            # 显示循环 FPS 统计
            disp_count += 1
            if current_time - last_disp_fps_time >= 10.0:
                disp_fps = disp_count / (current_time - last_disp_fps_time)
                print(f"[显示] FPS: {disp_fps:.1f}")
                with state_lock:
                    runtime_state["performance"]["display_fps"] = round(disp_fps, 1)
                disp_count = 0
                last_disp_fps_time = current_time

            # 限速到 30fps（推理线程独立运行，不受此限速影响）
            elapsed = time.time() - t_loop
            sleep_t = TARGET_DT - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("停止检测...")
    finally:
        running_flag[0] = False
        camera.release()
        influx.close()
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

if __name__ == '__main__':
    main()

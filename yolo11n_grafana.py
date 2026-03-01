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
        "inference_ms": 0.0
    }
}

def generate():
    global output_frame, frame_lock
    while True:
        with frame_lock:
            frame = None if output_frame is None else output_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype = "multipart/x-mixed-replace; boundary=frame")

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
        self.connect()

    def connect(self):
        try:
            self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
            # 异步写入，flush_interval=200ms 保证数据快速进库，batch_size=1 不等积累
            self.write_api = self.client.write_api(write_options=WriteOptions(
                batch_size=1,
                flush_interval=200,
            ))
            print(f"成功连接到 InfluxDB: {self.url}")
        except Exception as e:
            print(f"连接 InfluxDB 失败: {e}")
            self.client = None

    def send_person_count(self, count, timestamp=None):
        if self.write_api is None:
            return False
        try:
            point = Point("person_count") \
                .tag("source", "yolo11n_camera") \
                .tag("device", "jetson") \
                .field("count", int(count))
            if timestamp:
                point = point.time(timestamp)
            self.write_api.write(bucket=self.bucket, record=point)
            return True
        except Exception as e:
            print(f"发送数据失败: {e}")
            return False

    def send_all(self, count, avg_confidence, total_visitors, timestamp=None):
        """合并写入 person_count 和 detection_details，减少网络往返"""
        if self.write_api is None:
            return False
        try:
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
            self.write_api.write(bucket=self.bucket, record=[p1, p2])
            return True
        except Exception as e:
            print(f"发送数据失败: {e}")
            return False

    def send_detection_details(self, count, avg_confidence, timestamp=None):
        if self.write_api is None:
            return False
        try:
            point = Point("detection_details") \
                .tag("source", "yolo11n_camera") \
                .tag("device", "jetson") \
                .field("person_count", int(count)) \
                .field("avg_confidence", float(avg_confidence))
            if timestamp:
                point = point.time(timestamp)
            self.write_api.write(bucket=self.bucket, record=point)
            return True
        except Exception as e:
            print(f"发送详情失败: {e}")
            return False

    def send_uptime(self, uptime_seconds, start_time=None):
        if self.write_api is None:
            return False
        try:
            point = Point("app_status") \
                .tag("source", "yolo11n_camera") \
                .tag("device", "jetson") \
                .field("uptime_seconds", float(uptime_seconds))
            if start_time:
                point = point.field("start_time", float(start_time))
            self.write_api.write(bucket=self.bucket, record=point)
            return True
        except Exception as e:
            print(f"发送状态失败: {e}")
            return False

    def close(self):
        if self.client:
            self.client.close()

class HeatmapGenerator:
    def __init__(self, width, height, alpha=0.5, decay=0.95, ksize=51):
        self.width = width
        self.height = height
        self.heatmap = np.zeros((height, width), dtype=np.float32)
        self.alpha = alpha
        self.decay = decay
        if ksize % 2 == 0:
            ksize += 1
        g = cv2.getGaussianKernel(ksize, -1)
        kernel = g @ g.T
        kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-6)
        self.kernel = kernel.astype(np.float32)
        self.half = ksize // 2

    def update(self, centers):
        self.heatmap *= self.decay
        h = self.half
        k = self.kernel
        for x, y in centers:
            if 0 <= x < self.width and 0 <= y < self.height:
                x0 = max(0, x - h)
                y0 = max(0, y - h)
                x1 = min(self.width, x + h + 1)
                y1 = min(self.height, y + h + 1)
                kx0 = h - (x - x0)
                ky0 = h - (y - y0)
                kx1 = kx0 + (x1 - x0)
                ky1 = ky0 + (y1 - y0)
                self.heatmap[y0:y1, x0:x1] += k[ky0:ky1, kx0:kx1]
        np.clip(self.heatmap, 0, 1.0, out=self.heatmap)

    def apply_to_frame(self, frame):
        # 将热力图转换为彩色
        heatmap_norm = (self.heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        # 叠加
        return cv2.addWeighted(frame, 1 - self.alpha, heatmap_color, self.alpha, 0)

class USBCamera:
    def __init__(self, source=0, width=1280, height=720, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def open(self):
        # 检查 source 是否为数字（摄像头索引）或字符串（文件路径）
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"无法打开源: {self.source}")
            return False
            
        # 只有在是摄像头设备时才设置参数，视频文件通常不能设置FPS/分辨率
        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 缓冲区只保留1帧，始终读最新帧避免积压
        return True

    def read(self):
        if self.cap:
            return self.cap.read()
        return False, None

    def release(self):
        if self.cap:
            self.cap.release()

def main():
    parser = argparse.ArgumentParser(description='YOLO11n Person Detection with InfluxDB & Grafana')
    # 允许通过环境变量设置默认值
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
    
    args = parser.parse_args()

    # 初始化 InfluxDB
    influx = InfluxDBSender(args.influx_url, args.influx_token, args.influx_org, args.influx_bucket)

    # 初始化 YOLO 模型，优先使用 TensorRT engine
    if os.path.exists('/app/yolo11n.engine'):
        model_path = '/app/yolo11n.engine'
    elif os.path.exists('/app/yolo11n.pt'):
        model_path = '/app/yolo11n.pt'
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
    print(f"使用设备: {selected_device}")
    
    # 启动 Flask 视频流线程
    t = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=args.web_port, debug=False, use_reloader=False, threaded=True))
    t.daemon = True
    t.start()
    
    # 初始化摄像头/视频
    camera = USBCamera(args.source)
    if not camera.open():
        sys.exit(1)

    # 获取视频尺寸用于热力图
    width = int(camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    heatmap_gen = HeatmapGenerator(width, height, alpha=args.heatmap_alpha, ksize=args.heatmap_ksize)

    # 获取视频原始帧率，用于限速（视频文件才需要，摄像头自带帧率）
    src_fps = camera.cap.get(cv2.CAP_PROP_FPS)
    is_video_file = isinstance(args.source, str) and not args.source.isdigit()
    frame_duration = 1.0 / src_fps if (is_video_file and src_fps > 0) else 0.0
    print(f"视频帧率: {src_fps:.1f} fps，目标帧间隔: {frame_duration*1000:.1f}ms")

    print(f"开始检测: {args.source}")
    last_send_time = time.time()
    start_time = last_send_time
    last_status_send = last_send_time

    last_person_count = 0
    last_confidences = []
    last_centers = []
    last_frame = None

    # 今日累计人流
    total_visitors = 0
    prev_person_count = 0
    last_day = time.localtime().tm_mday

    try:
        frame_skip = 0
        frame_count = 0
        last_fps_time = time.time()
        while True:
            loop_start = time.time()
            ret, frame = camera.read()
            if not ret:
                print("视频播放结束或无法读取画面")
                # 如果是文件，循环播放
                if isinstance(args.source, str) and not args.source.isdigit():
                    print("重新播放视频...")
                    camera.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            # 跳帧处理：每2帧只做1次推理，提高处理速度
            frame_skip = (frame_skip + 1) % 2
            if frame_skip != 0:
                # 使用上一帧的结果
                person_count = last_person_count
                confidences = last_confidences.copy()
                centers = last_centers.copy()
            else:
                # 推理
                inference_start = time.time()
                results = model.predict(source=frame, classes=[0], verbose=False, device=selected_device, half=False, imgsz=640)
                inference_time = time.time() - inference_start
                
                person_count = 0
                confidences = []
                centers = []

                # 处理检测结果
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        person_count += 1
                        conf = float(box.conf[0])
                        confidences.append(conf)
                        
                        # 获取坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        centers.append((center_x, center_y))

                        # 模糊处理
                        if args.blur:
                            # 提取人像区域
                            face_roi = frame[y1:y2, x1:x2]
                            if face_roi.size > 0:
                                # 高斯模糊
                                face_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)
                                frame[y1:y2, x1:x2] = face_roi
                            
                            # 画框（可选，模糊后可能不需要框，这里画一个淡淡的框）
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 保存结果供跳帧使用
                last_person_count = person_count
                last_confidences = confidences.copy()
                last_centers = centers.copy()
                
                # 每10秒打印一次性能统计
                frame_count += 1
                if time.time() - last_fps_time >= 10:
                    fps = frame_count / 10.0
                    print(f"[性能] FPS: {fps:.1f}, 推理耗时: {inference_time*1000:.1f}ms, 人数: {person_count}")
                    with state_lock:
                        runtime_state["performance"]["fps"] = round(fps, 1)
                    frame_count = 0
                    last_fps_time = time.time()
            
            # 更新热力图
            heatmap_gen.update(centers)
            
            # 叠加热力图
            frame = heatmap_gen.apply_to_frame(frame)

            # 更新用于流媒体的帧
            global output_frame
            with frame_lock:
                output_frame = frame.copy()

            # 显示计数
            cv2.putText(frame, f"Person Count: {person_count}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 定时发送数据 (每200ms发送一次，与InfluxDB flush_interval匹配)
            current_time = time.time()
            if current_time - last_send_time >= 0.2:
                avg_conf = sum(confidences) / len(confidences) if confidences else 0
                write_timestamp = int(current_time * 1e9)

                # 累计今日人流：人数增加时计入新增量；每天零点重置
                current_day = time.localtime().tm_mday
                if current_day != last_day:
                    total_visitors = 0
                    last_day = current_day
                if person_count > prev_person_count:
                    total_visitors += (person_count - prev_person_count)
                prev_person_count = person_count

                write_start = time.time()
                influx.send_all(person_count, avg_conf, total_visitors, timestamp=write_timestamp)
                write_latency = (time.time() - write_start) * 1000

                last_send_time = current_time
                with state_lock:
                    runtime_state["ts"] = current_time
                    runtime_state["frame_idx"] = int(camera.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    runtime_state["source"] = str(args.source)
                    runtime_state["person_count"] = int(person_count)
                    runtime_state["total_visitors"] = int(total_visitors)
                    runtime_state["avg_confidence"] = float(avg_conf) if confidences else 0.0
                    runtime_state["confidences"] = [float(c) for c in confidences]
                    runtime_state["centers"] = [(int(x), int(y)) for (x, y) in centers]
                    runtime_state["influx"]["last_person_write_ts"] = float(last_send_time)
                    runtime_state["influx"]["bucket"] = args.influx_bucket
                    runtime_state["influx"]["org"] = args.influx_org
                    runtime_state["influx"]["write_latency_ms"] = round(write_latency, 2)
                    if frame_skip == 0:
                        runtime_state["performance"]["inference_ms"] = round(inference_time * 1000, 1)

            # 定时发送运行状态（每10秒）
            if current_time - last_status_send >= 10.0:
                uptime = current_time - start_time
                influx.send_uptime(uptime_seconds=uptime, start_time=start_time)
                last_status_send = current_time

            # 显示画面
            if not args.headless:
                try:
                    cv2.imshow('YOLO11n Person Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    print(f"无法显示画面 (可能是 headless 环境): {e}")

            # 帧率限速：视频文件按原始帧率播放，避免加速
            if frame_duration > 0:
                elapsed = time.time() - loop_start
                sleep_t = frame_duration - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("停止检测...")
    finally:
        camera.release()
        influx.close()
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

if __name__ == '__main__':
    main()

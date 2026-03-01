import cv2
import time
import argparse
import numpy as np
from ultralytics import YOLO
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import os
import sys
import threading
from flask import Flask, Response, jsonify
import torch
import json

app = Flask(__name__)
output_frame = None
frame_lock = threading.Lock()
state_lock = threading.Lock()
runtime_state = {
    "ts": 0.0,
    "frame_idx": 0,
    "source": "",
    "person_count": 0,
    "avg_confidence": 0.0,
    "confidences": [],
    "centers": [],
    "influx": {
        "last_person_write_ts": 0.0,
        "bucket": "",
        "org": ""
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
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + bytearray(encoded) + b'\r\n'

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

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
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            print(f"InfluxDB ok: {self.url}")
        except Exception as e:
            print(f"InfluxDB error: {e}")
            self.client = None

    def send_person_count(self, count, timestamp=None):
        if self.write_api is None:
            return False
        try:
            point = Point("person_count").tag("source", "yolo11n_camera").tag("device", "jetson").field("count", int(count))
            if timestamp:
                point = point.time(timestamp)
            self.write_api.write(bucket=self.bucket, record=point)
            return True
        except Exception as e:
            print(f"write person_count error: {e}")
            return False

    def send_detection_details(self, count, avg_confidence, timestamp=None):
        if self.write_api is None:
            return False
        try:
            point = Point("detection_details").tag("source", "yolo11n_camera").tag("device", "jetson").field("person_count", int(count)).field("avg_confidence", float(avg_confidence))
            if timestamp:
                point = point.time(timestamp)
            self.write_api.write(bucket=self.bucket, record=point)
            return True
        except Exception as e:
            print(f"write detection_details error: {e}")
            return False

    def send_uptime(self, uptime_seconds, start_time=None):
        if self.write_api is None:
            return False
        try:
            point = Point("app_status").tag("source", "yolo11n_camera").tag("device", "jetson").field("uptime_seconds", float(uptime_seconds))
            if start_time:
                point = point.field("start_time", float(start_time))
            self.write_api.write(bucket=self.bucket, record=point)
            return True
        except Exception as e:
            print(f"write app_status error: {e}")
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
        heatmap_norm = (self.heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 1 - self.alpha, heatmap_color, self.alpha, 0)

class USBCamera:
    def __init__(self, source=0, width=1280, height=720, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def open(self):
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"open source failed: {self.source}")
            return False
        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        return True

    def read(self):
        if self.cap:
            return self.cap.read()
        return False, None

    def release(self):
        if self.cap:
            self.cap.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=os.getenv("VIDEO_SOURCE", "0"))
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--influx-url', type=str, default=os.getenv("INFLUX_URL", "http://localhost:8086"))
    parser.add_argument('--influx-token', type=str, default=os.getenv("INFLUX_TOKEN", "my-super-secret-auth-token"))
    parser.add_argument('--influx-org', type=str, default=os.getenv("INFLUX_ORG", "jetson"))
    parser.add_argument('--influx-bucket', type=str, default=os.getenv("INFLUX_BUCKET", "person_detection"))
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--blur', action='store_true')
    parser.add_argument('--heatmap-alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default=os.getenv("DEVICE", "auto"))
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--heatmap-ksize', type=int, default=51)
    args = parser.parse_args()

    influx = InfluxDBSender(args.influx_url, args.influx_token, args.influx_org, args.influx_bucket)

    print("loading YOLO11n...")
    model = YOLO('yolo11n.pt')
    selected_device = args.device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif selected_device.isdigit():
        selected_device = int(selected_device)
    print(f"device: {selected_device}")
    if selected_device != "cpu":
        try:
            model.to('cuda')
        except Exception as e:
            print(f"cuda error: {e}")
            selected_device = "cpu"

    t = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True))
    t.daemon = True
    t.start()

    camera = USBCamera(args.source)
    if not camera.open():
        sys.exit(1)

    width = int(camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    heatmap_gen = HeatmapGenerator(width, height, alpha=args.heatmap_alpha, ksize=args.heatmap_ksize)

    print(f"start detect: {args.source}")
    last_send_time = time.time()
    start_time = last_send_time
    last_status_send = last_send_time

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("eof or read fail")
                if isinstance(args.source, str) and not args.source.isdigit():
                    camera.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            results = model.predict(source=frame, classes=[0], verbose=False, device=selected_device, half=(selected_device != "cpu"))
            person_count = 0
            confidences = []
            centers = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    person_count += 1
                    conf = float(box.conf[0])
                    confidences.append(conf)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    centers.append((center_x, center_y))
                    if args.blur:
                        roi = frame[y1:y2, x1:x2]
                        if roi.size > 0:
                            roi = cv2.GaussianBlur(roi, (99, 99), 30)
                            frame[y1:y2, x1:x2] = roi
            heatmap_gen.update(centers)
            frame = heatmap_gen.apply_to_frame(frame)
            global output_frame
            with frame_lock:
                output_frame = frame.copy()
            cv2.putText(frame, f"Person Count: {person_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            current_time = time.time()
            if current_time - last_send_time >= 1.0:
                avg_conf = sum(confidences) / len(confidences) if confidences else 0
                influx.send_person_count(person_count)
                if person_count > 0:
                    influx.send_detection_details(person_count, avg_conf)
                last_send_time = current_time
                with state_lock:
                    runtime_state["ts"] = current_time
                    runtime_state["frame_idx"] = int(camera.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    runtime_state["source"] = str(args.source)
                    runtime_state["person_count"] = int(person_count)
                    runtime_state["avg_confidence"] = float(avg_conf) if confidences else 0.0
                    runtime_state["confidences"] = [float(c) for c in confidences]
                    runtime_state["centers"] = [(int(x), int(y)) for (x, y) in centers]
                    runtime_state["influx"]["last_person_write_ts"] = float(last_send_time)
                    runtime_state["influx"]["bucket"] = args.influx_bucket
                    runtime_state["influx"]["org"] = args.influx_org
            if current_time - last_status_send >= 10.0:
                uptime = current_time - start_time
                influx.send_uptime(uptime_seconds=uptime, start_time=start_time)
                last_status_send = current_time
            if not args.headless:
                try:
                    cv2.imshow('YOLO11n Person Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception:
                    pass
    except KeyboardInterrupt:
        print("stop")
    finally:
        camera.release()
        influx.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

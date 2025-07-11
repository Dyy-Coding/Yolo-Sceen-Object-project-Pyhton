from flask import Flask, render_template, Response, jsonify, request
import cv2
import torch
from ultralytics import YOLO
from collections import Counter
import random
import time
import threading
import numpy as np

# pip install -r requirements.txt


app = Flask(__name__)

model = YOLO('yolov8s.pt')
model.fuse()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

object_counts = {}
fps = 0.0
scanning = False
lock = threading.Lock()
object_colors = {}

def get_color(key):
    if key not in object_colors:
        object_colors[key] = [random.randint(0, 255) for _ in range(3)]
    return object_colors[key]

cap = cv2.VideoCapture(0)

def generate_frames():
    global object_counts, fps
    while True:
        with lock:
            is_scanning = scanning

        if not is_scanning:
            time.sleep(0.1)
            # yield black frame to keep connection alive (optional)
            black_frame = 255 * np.zeros((480, 640, 3), dtype='uint8')
            _, buffer = cv2.imencode('.jpg', black_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue

        success, frame = cap.read()
        if not success:
            break

        start_time = time.time()
        results = model.track(frame, persist=True, conf=0.5)
        temp_counts = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                conf = box.conf.cpu().item()
                if conf < 0.5:
                    continue

                cls_id = int(box.cls.cpu().item())
                obj_name = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
                track_id = int(box.id.cpu().item()) if hasattr(box, 'id') and box.id is not None else -1
                color = get_color(track_id if track_id != -1 else obj_name)
                label = f"{obj_name} [ID: {track_id}] ({conf:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                temp_counts.append(obj_name)

        with lock:
            object_counts = dict(Counter(temp_counts))
            fps = 1.0 / (time.time() - start_time + 1e-6)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # Your frontend HTML here

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    with lock:
        return jsonify({
            "object_counts": object_counts,
            "fps": round(fps, 2)
        })

@app.route('/start', methods=['POST'])
def start_scan():
    global scanning
    with lock:
        scanning = True
    return jsonify({"success": True, "message": "Scanning started."})

@app.route('/stop', methods=['POST'])
def stop_scan():
    global scanning, object_counts, fps
    with lock:
        scanning = False
        object_counts = {}
        fps = 0.0
    return jsonify({"success": True, "message": "Scanning stopped."})

if __name__ == '__main__':
    app.run(debug=True)

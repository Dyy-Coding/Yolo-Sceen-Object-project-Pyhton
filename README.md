# 📸 Real-Time Object Detection Web App  
### YOLOv8 + Flask | Gen Z Edition

---

## 🧠 Overview

This project is a fun, fast, and efficient real-time object detection app powered by:

- 🧪 **Flask** (backend server & REST API)
- 🧠 **YOLOv8** (Ultralytics object detection & tracking)
- 🎥 **OpenCV** (webcam video stream)
- 🚀 **Live Web UI** (browser-based viewer)

Perfect for demos, prototypes, or building your own AI cam 🧃.

---

## 🔥 Features

- 🖥️ Stream webcam to browser in real time
- 📦 Object detection + tracking with bounding boxes
- 📊 Live object counts and FPS display
- 🕹️ Start/stop scanning from browser or via API
- 👾 Easily customizable and Gen Z–friendly interface

---

## 🧰 Requirements

- Python 3.8+
- PyTorch (with GPU/CUDA if available)
- Webcam (USB or built-in)

---

## 📦 Installation

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/flask-yolo-object-tracker.git
cd flask-yolo-object-tracker
```

### 2. Create a Virtual Environment (optional but 🔥)
```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Run the App

```bash
python app.py
```

Then open your browser at:  
📍 [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🔌 API Endpoints

| Method | Route         | Description                        |
|--------|---------------|------------------------------------|
| GET    | `/`           | Web interface                      |
| GET    | `/video_feed` | MJPEG webcam stream                |
| GET    | `/status`     | JSON with object counts & FPS      |
| POST   | `/start`      | Starts object detection            |
| POST   | `/stop`       | Stops object detection             |

---

## 🎯 YOLOv8 Model

This project uses the small model by default:
```python
model = YOLO('yolov8s.pt')
```

You can replace it with:
- `yolov8n.pt` (nano)
- `yolov8m.pt` (medium)
- `yolov8l.pt` (large)

---

## 🗂️ Project Structure

```
📦 flask-yolo-object-tracker
├── app.py              # Main Flask app
├── templates/
│   └── index.html      # UI for live video and controls
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## ✨ Custom Ideas

- 🎛 Filter detections by class (e.g., only show people or cars)
- 💾 Add snapshot download button in UI
- 🌙 Add dark/light mode toggle
- 📈 Store detection logs for analytics

---

## 📃 License

MIT License  
Feel free to fork, remix, and reuse. Just give a shoutout if you build something awesome 🚀

---

## 🏁 Let's Build Something Cool

> AI-powered vision from your own laptop — no cloud needed.  
> Real-time insights + real Gen Z style. 💻⚡🧃

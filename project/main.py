import cv2
import torch
from ultralytics import YOLO
from collections import Counter
import random

# Load YOLOv8 model (pre-trained)
model = YOLO('yolov8s.pt')  # Use YOLOv8 Small for speed and reasonable accuracy

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Access webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Confidence threshold and target objects
confidence_threshold = 0.5

# Assign random colors for each object type
object_colors = {}

def get_object_color(obj_name):
    if obj_name not in object_colors:
        object_colors[obj_name] = [random.randint(0, 255) for _ in range(3)]
    return object_colors[obj_name]

# Assign a unique trainID for detected objects
trainID_mapping = {}
next_trainID = 0

def get_trainID(obj_name):
    global next_trainID
    if obj_name not in trainID_mapping:
        trainID_mapping[obj_name] = next_trainID
        next_trainID += 1
    return trainID_mapping[obj_name]

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform object detection
    results = model(frame)  # Results on the current frame

    # Filter detections by confidence
    detections = results[0].boxes  # YOLOv8 results (Boxes object)
    objects = []
    for detection in detections:
        conf = detection.conf.cpu().item()
        if conf > confidence_threshold:
            cls = int(detection.cls.cpu().item())  # Class index
            x1, y1, x2, y2 = map(int, detection.xyxy[0].cpu().tolist())  # Bounding box
            obj_name = model.names[cls]  # Object name from model
            objects.append(obj_name)

            # Annotate frame
            color = get_object_color(obj_name)  # Get color for the object
            trainID = get_trainID(obj_name)     # Get trainID for the object
            label = f"{obj_name} ({conf:.2f}) [ID: {trainID}]"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Count detected objects
    object_counts = Counter(objects)

    # Display object counts
    y_offset = 20
    cv2.putText(frame, "Detected objects:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 20
    for obj, count in object_counts.items():
        trainID = get_trainID(obj)
        text = f"{obj} (ID: {trainID}): {count}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 20

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

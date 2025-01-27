import cv2
import torch
from collections import Counter
import random

# Load YOLOv5 model (pre-trained)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

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
    results = model(frame)
    detections = results.pandas().xyxy[0]  # Extract detection results as a pandas DataFrame

    # Filter detections by confidence
    detections = detections[detections['confidence'] > confidence_threshold]

    # Parse detected objects
    objects = detections['name'].tolist()  # Extract object names
    object_counts = Counter(objects)       # Count objects

    # Annotate frame with detected objects
    for index, row in detections.iterrows():
        x1, y1, x2, y2, conf, cls, name = row
        color = get_object_color(name)  # Get color for the object
        trainID = get_trainID(name)     # Get trainID for the object
        label = f"{name} ({conf:.2f}) [ID: {trainID}]"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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

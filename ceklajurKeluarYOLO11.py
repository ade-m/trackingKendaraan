import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque
import sys
sys.path.append("sort")
from sort import Sort

# Load model YOLOv11
model = YOLO("yolo11n.pt")

# Buka video ATCS
video_path = "https://atcsdishub.pemkomedan.go.id/camera/TVRI.m3u8"
cap = cv2.VideoCapture(video_path)

# Inisialisasi tracker SORT
tracker = Sort(max_age=50, min_hits=5, iou_threshold=0.2)

# Dictionary untuk menyimpan jejak pergerakan kendaraan dan warna per ID
tracks = {}
colors = {}

# Inisialisasi penghitung kendaraan berdasarkan lajur
counts = {"left": 0, "straight": 0, "right": 0}
counted_objects = {}

# Garis pembatas area keluar dalam bentuk kotak
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
box_left = (50, 450, 200, frame_height - 25)
box_right = (frame_width - 200, 300, frame_width - 50, frame_height - 25)
horizontal_box = (frame_width // 3, frame_height - 100, frame_width * 2 // 3, frame_height - 10)

# Fungsi untuk mendapatkan warna unik berdasarkan ID
np.random.seed(42)
def get_color(id):
    if id not in colors:
        colors[id] = tuple(map(int, np.random.randint(0, 255, size=3)))
    return colors[id]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    detections = [[*map(int, box.xyxy[0]), box.conf[0].item()] 
                  for result in results 
                  for box in result.boxes 
                  if int(box.cls[0].item()) in [2, 3, 5, 7]]
    
    tracked_objects = tracker.update(np.array(detections))
    
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        color = get_color(obj_id)

        if obj_id not in tracks:
            tracks[obj_id] = deque(maxlen=30)
        tracks[obj_id].append((cx, cy))
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if obj_id not in counted_objects:
            if box_left[0] < cx < box_left[2] and box_left[1] < cy < box_left[3]:
                counts["left"] += 1
                counted_objects[obj_id] = "left"
            elif box_right[0] < cx < box_right[2] and box_right[1] < cy < box_right[3]:
                counts["right"] += 1
                counted_objects[obj_id] = "right"
            elif horizontal_box[0] < cx < horizontal_box[2] and horizontal_box[1] < cy < horizontal_box[3]:
                counts["straight"] += 1
                counted_objects[obj_id] = "straight"
    
    cv2.rectangle(frame, (box_left[0], box_left[1]), (box_left[2], box_left[3]), (0, 0, 255), 2)
    cv2.rectangle(frame, (box_right[0], box_right[1]), (box_right[2], box_right[3]), (0, 255, 0), 2)
    cv2.rectangle(frame, (horizontal_box[0], horizontal_box[1]), (horizontal_box[2], horizontal_box[3]), (255, 0, 0), 2)
    
    cv2.putText(frame, "Kiri :"+str(counts['left']), (box_left[0] + 20, box_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, "Kanan:"+str(counts['right']), (box_right[0] - 80, box_right[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, "Lurus:"+str(counts['straight']), (horizontal_box[0] + 20, horizontal_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    for obj_id, points in tracks.items():
        for i in range(1, len(points)):
            if points[i - 1] and points[i]:
                cv2.line(frame, points[i - 1], points[i], get_color(obj_id), 2)
    
    cv2.imshow("Tracking Kendaraan", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Kendaraan keluar ke kiri: {counts['left']}")
print(f"Kendaraan keluar ke kanan: {counts['right']}")
print(f"Kendaraan keluar lurus: {counts['straight']}")

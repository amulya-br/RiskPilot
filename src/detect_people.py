from ultralytics import YOLO
import cv2

# Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Open webcam (0) or replace 0 with a video file path
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame,conf=0.5)

    annotated_frame = results[0].plot()

    cv2.imshow("RiskPilot - Person Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
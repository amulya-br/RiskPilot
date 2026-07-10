from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Track people
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],      # Only detect persons
        conf=0.5
    )

    annotated_frame = results[0].plot()

    # Count tracked persons
    count = 0

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy()
        count = len(ids)

    cv2.putText(
        annotated_frame,
        f"People Count: {count}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("RiskPilot - Person Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
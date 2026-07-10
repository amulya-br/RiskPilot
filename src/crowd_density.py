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

    # Detect only persons
    results = model(frame, classes=[0], conf=0.5)

    annotated_frame = frame.copy()

    person_count = 0

    for box in results[0].boxes:
        person_count += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255,0,0), 2)

        cv2.putText(
            annotated_frame,
            "Person",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,0,0),
            2
        )

    # Density Analysis
    if person_count <= 2:
        density = "LOW"
        risk = "SAFE"
        color = (0,255,0)

    elif person_count <= 5:
        density = "MEDIUM"
        risk = "MODERATE"
        color = (0,255,255)

    else:
        density = "HIGH"
        risk = "HIGH"
        color = (0,0,255)

    # Display information
    cv2.putText(
        annotated_frame,
        f"People Count : {person_count}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.putText(
        annotated_frame,
        f"Density : {density}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.putText(
        annotated_frame,
        f"Risk Level : {risk}",
        (20,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("RiskPilot - Crowd Density Analysis", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
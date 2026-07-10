from ultralytics import YOLO
import cv2
import math

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Store previous center points
previous_positions = {}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.5,
        verbose=False
    )

    annotated_frame = frame.copy()

    moving = 0
    stationary = 0
    person_count = 0

    if results[0].boxes.id is not None:

        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        person_count = len(ids)

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Draw bounding box
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                f"ID:{track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

            # Check movement
            if track_id in previous_positions:

                px, py = previous_positions[track_id]

                distance = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)

                # Smaller threshold for webcam
                if distance > 3:
                    moving += 1
                else:
                    stationary += 1

            else:
                stationary += 1

            previous_positions[track_id] = (cx, cy)

    # Density
    if person_count <= 2:
        density = "LOW"
        risk = "SAFE"
        color = (0, 255, 0)

    elif person_count <= 5:
        density = "MEDIUM"
        risk = "MODERATE"
        color = (0, 255, 255)

    else:
        density = "HIGH"
        risk = "HIGH"
        color = (0, 0, 255)

    # Movement Status
    if moving >= max(1, person_count // 2):
        movement_status = "ACTIVE"
    else:
        movement_status = "NORMAL"

    # Display
    cv2.putText(
        annotated_frame,
        f"People Count : {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )

    cv2.putText(
        annotated_frame,
        f"Density : {density}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )

    cv2.putText(
        annotated_frame,
        f"Moving : {moving}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )

    cv2.putText(
        annotated_frame,
        f"Stationary : {stationary}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )

    cv2.putText(
        annotated_frame,
        f"Status : {movement_status}",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )

    cv2.putText(
        annotated_frame,
        f"Risk : {risk}",
        (20, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )

    cv2.imshow("RiskPilot - Movement Analysis", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
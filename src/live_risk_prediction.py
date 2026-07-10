import cv2
import joblib
import numpy as np
from ultralytics import YOLO
import math
from live_data import dashboard_data
# -----------------------------
# Load YOLO Model
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# Load Random Forest Model
# -----------------------------
risk_model = joblib.load("models/risk_model.pkl")

# -----------------------------
# Open Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Store previous positions
previous_positions = {}

movement_threshold = 3

print("RiskPilot Started...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        verbose=False
    )

    annotated = results[0].plot()

    people_count = 0

    if results[0].boxes is not None:
        people_count = len(results[0].boxes)

    # -----------------------------
    # Crowd Density
    # -----------------------------
    if people_count <= 5:
        density = 0
        density_name = "Low"

    elif people_count <= 10:
        density = 1
        density_name = "Medium"

    else:
        density = 2
        density_name = "High"

    # -----------------------------
    # Live Movement Analysis
    # -----------------------------
    moving = 0
    stationary = 0
    total_speed = 0

    if (
        results[0].boxes is not None
        and results[0].boxes.id is not None
    ):

        ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().tolist()

        for track_id, box in zip(ids, boxes):

            x1, y1, x2, y2 = box

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if track_id in previous_positions:

                px, py = previous_positions[track_id]

                speed = math.sqrt(
                    (cx - px) ** 2 +
                    (cy - py) ** 2
                )

                total_speed += speed

                if speed > movement_threshold:
                    moving += 1
                else:
                    stationary += 1

            else:
                stationary += 1

            previous_positions[track_id] = (cx, cy)

    if people_count > 0:
        avg_speed = total_speed / people_count
    else:
        avg_speed = 0

    # -----------------------------
    # AI Prediction
    # -----------------------------
    X = np.array([[
        people_count,
        density,
        moving,
        stationary,
        avg_speed
    ]])

    prediction = risk_model.predict(X)[0]
    dashboard_data["people"] = people_count
    dashboard_data["density"] = density_name
    dashboard_data["moving"] = moving
    dashboard_data["stationary"] = stationary
    dashboard_data["speed"] = round(avg_speed, 2)
    dashboard_data["risk"] = prediction.upper()

    # -----------------------------
    # Risk Color
    # -----------------------------
    if prediction == "Safe":
        risk_color = (0,255,0)

    elif prediction == "Moderate":
        risk_color = (0,255,255)

    elif prediction == "High":
        risk_color = (0,165,255)

    else:
        risk_color = (0,0,255)

    # -----------------------------
    # Dashboard
    # -----------------------------
    cv2.rectangle(annotated,(10,10),(360,240),(0,0,0),-1)

    cv2.putText(
        annotated,
        "RiskPilot AI",
        (20,35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    cv2.putText(
        annotated,
        f"People : {people_count}",
        (20,70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        annotated,
        f"Density : {density_name}",
        (20,100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        annotated,
        f"Moving : {moving}",
        (20,130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        annotated,
        f"Stationary : {stationary}",
        (20,160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        annotated,
        f"Avg Speed : {avg_speed:.2f}",
        (20,190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        annotated,
        f"Risk : {prediction.upper()}",
        (20,225),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        risk_color,
        3
    )

    cv2.imshow("RiskPilot - Live AI", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
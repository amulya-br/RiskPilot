import cv2
import joblib
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
import math

# -------------------------
# Load Models
# -------------------------

yolo = YOLO("yolov8n.pt")

risk_model = joblib.load("models/risk_model.pkl")

# -------------------------
# Dataset Path
# -------------------------

image_folder = "data/MOT17/train/MOT17-02-FRCNN/img1"

track_history = defaultdict(list)

# -------------------------
# Process Frames
# -------------------------

for frame_num in range(1, 601, 30):

    image_path = f"{image_folder}/{frame_num:06d}.jpg"

    frame = cv2.imread(image_path)

    if frame is None:
        continue

    results = yolo.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        verbose=False,
    )

    annotated = results[0].plot()

    people_count = 0
    moving = 0
    stationary = 0
    total_speed = 0

    if (
        results[0].boxes is not None
        and results[0].boxes.id is not None
    ):

        ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().tolist()

        people_count = len(ids)

        for track_id, box in zip(ids, boxes):

            x1, y1, x2, y2 = box

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            history = track_history[track_id]

            history.append((cx, cy))

            if len(history) >= 2:

                dx = history[-1][0] - history[-2][0]
                dy = history[-1][1] - history[-2][1]

                speed = math.sqrt(dx * dx + dy * dy)

                total_speed += speed

                if speed > 5:
                    moving += 1
                else:
                    stationary += 1

    # -------------------------
    # Density
    # -------------------------

    if people_count <= 5:
        density = 0
        density_name = "Low"

    elif people_count <= 15:
        density = 1
        density_name = "Medium"

    else:
        density = 2
        density_name = "High"

    # -------------------------
    # Average Speed
    # -------------------------

    if people_count > 0:
        avg_speed = total_speed / people_count
    else:
        avg_speed = 0

    # -------------------------
    # Predict Risk
    # -------------------------

    X = pd.DataFrame(
        [[
            people_count,
            density,
            moving,
            stationary,
            avg_speed
        ]],
        columns=[
            "people_count",
            "density",
            "moving_people",
            "stationary_people",
            "avg_speed"
        ]
    )

    prediction = risk_model.predict(X)[0]

    # -------------------------
    # Display
    # -------------------------
    if prediction == "Safe":
        risk_color = (0, 255, 0)

    elif prediction == "Moderate":
        risk_color = (0, 255, 255)

    elif prediction == "High":
        risk_color = (0, 165, 255)

    else:
        risk_color = (0, 0, 255)
    
    # ---------------- Display ----------------

    cv2.putText(
        annotated,
        "RiskPilot AI",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        )

    cv2.putText(
        annotated,
        f"Frame : {frame_num}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2,
    )
    cv2.putText(
        annotated,
        f"People : {people_count}",
        (20,110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2,
    )

    cv2.putText(
        annotated,
        f"Density : {density_name}",
        (20,150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2,
    )
    cv2.putText(
        annotated,
        f"Moving : {moving}",
        (20,190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2,
    )
    cv2.putText(
        annotated,
        f"Avg Speed : {avg_speed:.2f}",
        (20,230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2,
    )
    cv2.putText(
        annotated,
        f"Risk : {prediction.upper()}",
        (20,280),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        risk_color,
        3,
    )
    cv2.imshow("RiskPilot - Dataset Mode", annotated)

    key = cv2.waitKey(700)

    if key == ord("q"):
        break

cv2.destroyAllWindows()
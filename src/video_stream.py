import cv2
import csv
import os
import math
import joblib
import numpy as np
from datetime import datetime
from ultralytics import YOLO

from live_data import dashboard_data

# ==========================================
# RiskPilot AI
# ==========================================

model = YOLO("yolov8m.pt")

risk_model = joblib.load("models/risk_model.pkl")

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

VIDEO_FILE = os.path.join(
    BASE_DIR,
    "static",
    "videos",
    "crowd.mp4"
)

VIDEO_MODE = "camera"

cap = None

previous_positions = {}

movement_threshold = 3

# ==========================================
# CSV Logging
# ==========================================

os.makedirs("logs", exist_ok=True)

csv_file = os.path.join(
    "logs",
    "risk_log.csv"
)

if not os.path.exists(csv_file):

    with open(csv_file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Time",
            "People",
            "Density",
            "Moving",
            "Stationary",
            "Speed",
            "Risk"
        ])

last_logged_second = -1


# ==========================================
# Open Video
# ==========================================

def open_video():

    global cap

    if cap is not None:

        cap.release()

    if VIDEO_MODE == "camera":

        print("\nOpening Live Camera...")

        cap = cv2.VideoCapture(0)

    else:

        print("\nOpening Dataset Video...")

        cap = cv2.VideoCapture(VIDEO_FILE)

    if not cap.isOpened():

        raise Exception("Cannot Open Video Source")

    print("Video Ready")

open_video()
# ==========================================
# Generate Frames
# ==========================================

def generate_frames():

    global cap
    global VIDEO_MODE
    global previous_positions
    global last_logged_second

    while True:

        # -----------------------------
        # Read Frame
        # -----------------------------

        success, frame = cap.read()

        # Restart dataset automatically

        if not success:

            if VIDEO_MODE == "dataset":

                cap.release()

                cap = cv2.VideoCapture(VIDEO_FILE)

                continue

            else:

                cap.release()

                cap = cv2.VideoCapture(0)

                continue

        # -----------------------------
        # YOLO Detection
        # -----------------------------

        results = model.track(

            frame,

            persist=True,

            tracker="bytetrack.yaml",

            classes=[0],
            conf=0.20,
            iou=0.45,
            imgsz=1280,
            verbose=False

        )

        annotated = frame.copy()

        people_count = 0
        moving = 0
        stationary = 0
        total_speed = 0

        # -----------------------------
        # Count People
        # -----------------------------

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
        # Movement Analysis
        # -----------------------------

        if (

            results[0].boxes is not None

            and

            results[0].boxes.id is not None

        ):

            ids = results[0].boxes.id.int().cpu().tolist()

            boxes = results[0].boxes.xyxy.cpu().tolist()

            for track_id, box in zip(ids, boxes):

                x1, y1, x2, y2 = map(int, box)

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

        # -----------------------------
        # Average Speed
        # -----------------------------

        if people_count > 0:

            avg_speed = round(

                total_speed / people_count,

                2

            )

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

        prediction = risk_model.predict(X)[0].upper()
                # ==========================================
        # Dashboard Update
        # ==========================================

        dashboard_data["people"] = people_count
        dashboard_data["density"] = density_name
        dashboard_data["moving"] = moving
        dashboard_data["stationary"] = stationary
        dashboard_data["speed"] = round(avg_speed, 2)
        dashboard_data["risk"] = prediction

        # ==========================================
        # Alert Message
        # ==========================================

        if prediction == "SAFE":

            dashboard_data["alert"] = "Normal Activity"

            box_color = (0,255,0)

        elif prediction == "MODERATE":

            dashboard_data["alert"] = "Monitor Crowd"

            box_color = (0,255,255)

        elif prediction == "HIGH":

            dashboard_data["alert"] = "High Crowd Risk"

            box_color = (0,165,255)

        else:

            dashboard_data["alert"] = "Critical Situation"

            box_color = (0,0,255)

        # ==========================================
        # CSV Logging
        # ==========================================

        current_second = datetime.now().second

        if current_second != last_logged_second:

            last_logged_second = current_second

            with open(csv_file, "a", newline="") as file:

                writer = csv.writer(file)

                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    people_count,
                    density_name,
                    moving,
                    stationary,
                    round(avg_speed,2),
                    prediction
                ])

        # ==========================================
        # Draw Bounding Boxes
        # ==========================================

        if (
            results[0].boxes is not None
            and
            results[0].boxes.id is not None
        ):

            ids = results[0].boxes.id.int().cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().tolist()

            for track_id, box in zip(ids, boxes):

                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )

                cv2.putText(
                    annotated,
                    f"ID {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    box_color,
                    2
                )

        # ==========================================
        # Dashboard Overlay
        # ==========================================

        cv2.rectangle(
            annotated,
            (10,10),
            (360,230),
            (30,30,30),
            -1
        )

        cv2.putText(
            annotated,
            "RiskPilot AI",
            (20,35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255,255,255),
            2
        )

        cv2.putText(
            annotated,
            f"People : {people_count}",
            (20,70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255,255,255),
            2
        )

        cv2.putText(
            annotated,
            f"Density : {density_name}",
            (20,100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255,255,255),
            2
        )

        cv2.putText(
            annotated,
            f"Moving : {moving}",
            (20,130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255,255,255),
            2
        )

        cv2.putText(
            annotated,
            f"Stationary : {stationary}",
            (20,160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255,255,255),
            2
        )

        cv2.putText(
            annotated,
            f"Speed : {avg_speed:.2f}",
            (20,190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255,255,255),
            2
        )

        cv2.putText(
            annotated,
            f"Risk : {prediction}",
            (20,220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            box_color,
            2
        )

        # ==========================================
        # Encode Frame
        # ==========================================

        ret, buffer = cv2.imencode(".jpg", annotated)

        if not ret:

            continue

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame_bytes +
            b'\r\n'
        )
        # ==========================================
# Change Video Mode
# ==========================================

def set_video_mode(mode):

    global VIDEO_MODE
    global cap
    global previous_positions

    if mode not in ["camera", "dataset"]:
        return

    # Do nothing if already in same mode
    if VIDEO_MODE == mode:
        return

    print("=" * 60)
    print(f"Switching to {mode.upper()} Mode")
    print("=" * 60)

    VIDEO_MODE = mode

    previous_positions.clear()

    if cap is not None:
        cap.release()

    open_video()


# ==========================================
# Release Camera
# ==========================================

def release_camera():

    global cap

    if cap is not None:

        cap.release()

    cv2.destroyAllWindows()


# ==========================================
# Exit Information
# ==========================================

print("=" * 60)
print("RiskPilot AI Video Stream Ready")
print("Current Mode :", VIDEO_MODE.upper())
print("Dataset :", VIDEO_FILE)
print("=" * 60)
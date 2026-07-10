import cv2
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
import math
import os

# ======================================================
# RiskPilot AI - Professional Dataset Generator
# ======================================================

print("=" * 70)
print("RiskPilot AI Dataset Generator")
print("=" * 70)

# -----------------------------
# Load YOLOv8 Model
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# MOT17 Sequences
# -----------------------------
folders = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN"
]

rows = []

print(f"Total Sequences : {len(folders)}")
print()

# ======================================================
# Process Every Sequence
# ======================================================

for folder in folders:

    print(f"Processing : {folder}")

    image_folder = f"data/MOT17/train/{folder}/img1"

    if not os.path.exists(image_folder):

        print(f"Folder Missing : {image_folder}")
        print()
        continue

    # Store previous positions
    track_history = defaultdict(list)

    # --------------------------------------------------
    # Process every 10th frame
    # (more samples than every 30th frame)
    # --------------------------------------------------

    for frame_num in range(1, 601, 10):

        image_path = f"{image_folder}/{frame_num:06d}.jpg"

        frame = cv2.imread(image_path)

        if frame is None:
            continue

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            verbose=False
        )

        people_count = 0
        moving = 0
        stationary = 0
        total_speed = 0
                # ======================================================
        # Person Tracking
        # ======================================================

        if (
            results[0].boxes is not None
            and results[0].boxes.id is not None
        ):

            ids = results[0].boxes.id.int().cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().tolist()

            people_count = len(ids)

            for track_id, box in zip(ids, boxes):

                x1, y1, x2, y2 = box

                # Person Center
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Save position history
                history = track_history[track_id]
                history.append((cx, cy))

                # Keep last 15 positions only
                if len(history) > 15:
                    history.pop(0)

                # -----------------------------
                # Speed Calculation
                # -----------------------------
                if len(history) >= 2:

                    px, py = history[-2]

                    dx = cx - px
                    dy = cy - py

                    speed = math.sqrt(dx**2 + dy**2)

                    total_speed += speed

                    # Movement Detection
                    if speed >= 5:

                        moving += 1

                    else:

                        stationary += 1

                else:

                    stationary += 1

        # ======================================================
        # Average Speed
        # ======================================================

        if people_count > 0:

            avg_speed = round(
                total_speed / people_count,
                2
            )

        else:

            avg_speed = 0
                    # ======================================================
        # Crowd Density Classification
        # ======================================================

        if people_count <= 4:

            density = 0
            density_name = "Low"

        elif people_count <= 9:

            density = 1
            density_name = "Medium"

        else:

            density = 2
            density_name = "High"

        # ======================================================
        # Smart Risk Classification
        # ======================================================

        if people_count <= 4:

            # Small Crowd
            if avg_speed >= 20:
                risk = "Moderate"
            else:
                risk = "Safe"

        elif people_count <= 9:

            # Medium Crowd
            if moving >= (people_count * 0.70):
                risk = "Moderate"

            elif avg_speed >= 25:
                risk = "High"

            else:
                risk = "Moderate"

        elif people_count <= 15:

            # Dense Crowd
            if moving >= (people_count * 0.80):

                risk = "High"

            elif avg_speed >= 30:

                risk = "High"

            else:

                risk = "Moderate"

        else:

            # Very Dense Crowd

            if people_count >= 20:

                risk = "Critical"

            elif moving >= (people_count * 0.90):

                risk = "Critical"

            elif avg_speed >= 35:

                risk = "Critical"

            else:

                risk = "High"

        # ======================================================
        # Save Sample
        # ======================================================

        rows.append(

            [

                folder,
                frame_num,
                people_count,
                density,
                density_name,
                moving,
                stationary,
                avg_speed,
                risk

            ]

        )
        # ======================================================
# Create DataFrame
# ======================================================

df = pd.DataFrame(
    rows,
    columns=[
        "sequence",
        "frame",
        "people_count",
        "density",
        "density_name",
        "moving_people",
        "stationary_people",
        "avg_speed",
        "risk"
    ]
)

# ======================================================
# Sort Dataset
# ======================================================

df = df.sort_values(
    by=["sequence", "frame"]
).reset_index(drop=True)

# ======================================================
# Create data folder
# ======================================================

os.makedirs("data", exist_ok=True)

dataset_path = "data/risk_dataset.csv"

# ======================================================
# Save Dataset
# ======================================================

df.to_csv(
    dataset_path,
    index=False
)

# ======================================================
# Dataset Statistics
# ======================================================

print("\n" + "=" * 70)
print("DATASET GENERATED SUCCESSFULLY")
print("=" * 70)

print(f"\nSaved File : {dataset_path}")

print(f"Total Samples : {len(df)}")

print("\nRisk Distribution")
print("-" * 40)
print(df["risk"].value_counts())

print("\nDensity Distribution")
print("-" * 40)
print(df["density_name"].value_counts())

print("\nAverage People Count")
print("-" * 40)
print(round(df["people_count"].mean(), 2))

print("\nAverage Speed")
print("-" * 40)
print(round(df["avg_speed"].mean(), 2))

print("\nSample Dataset")
print("-" * 40)
print(df.head(10))

print("\n" + "=" * 70)
print("RiskPilot AI Dataset Generation Completed")
print("=" * 70)
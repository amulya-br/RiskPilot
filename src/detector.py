from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect(frame):
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        verbose=False
    )

    return results
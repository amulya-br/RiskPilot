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

    # Run YOLO detection
    results = model(frame, conf=0.5)

    # Copy original frame
    annotated_frame = frame.copy()

    # Initialize person count
    person_count = 0

    # Process detections
    for box in results[0].boxes:
        class_id = int(box.cls[0])

        # Detect only persons
        if model.names[class_id] == "person":
            person_count += 1

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Display label
            cv2.putText(
                annotated_frame,
                "Person",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

    # Display people count
    cv2.putText(
        annotated_frame,
        f"People Count: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Show output
    cv2.imshow("RiskPilot - Person Counter", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
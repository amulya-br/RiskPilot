import os
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

video = os.path.join(BASE_DIR, "static", "videos", "crowd.mp4")

print("BASE_DIR =", BASE_DIR)
print("VIDEO =", video)
print("Exists =", os.path.exists(video))

cap = cv2.VideoCapture(video)

print("Opened =", cap.isOpened())

ret, frame = cap.read()

print("First Frame =", ret)

cap.release()
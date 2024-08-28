# Resize the video to 640x480
import cv2
import numpy as np

cap = cv2.VideoCapture("output/video/compiled_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/video/compiled_video_resized.mp4', fourcc, 25.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    resized_frame = cv2.resize(frame, (640, 480))
    out.write(resized_frame)

cap.release()
out.release()
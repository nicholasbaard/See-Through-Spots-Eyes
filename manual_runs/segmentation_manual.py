import cv2
import numpy as np
import random
from PIL import Image
from ultralytics import YOLO
import os

# Initialize YOLOv8 model for person detection
model = YOLO("../models/yolov8n-seg.pt")  # Use 'yolov8n-seg.pt' for segmentation
model.to("cpu")
class_names = model.names
print('Class Names:', class_names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to 640x480
        frame = cv2.resize(frame, (640, 480))
        image = frame.copy()

        #segmentation

        results = model.predict(image, conf=0.5, verbose = False)
        image = results[0].plot()
        # Write the processed frame
        out.write(image)

    # Release resources
    cap.release()
    out.release()

if __name__ == '__main__':
    input_path = "../data_collection/output/video/compiled_video.mp4"
    output_path = "../data_collection/output/video/compiled_video_segmentation.mp4"



    process_video(input_path, output_path)
    print(f"Processed video saved as {output_path}")

    print("All videos processed.")

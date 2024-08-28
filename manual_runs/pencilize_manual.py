import cv2
import numpy as np
import random
from PIL import Image
from ultralytics import YOLO
import os

# Define the cartoon filter
def apply_cartoon_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inverted_gray = cv2.bitwise_not(gray)
    sketch = cv2.GaussianBlur(inverted_gray, (21, 21), sigmaX=0, sigmaY=0)
    sketch = cv2.bitwise_not(sketch)
    return cv2.divide(gray, sketch, scale=256.0)

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
    from tqdm import tqdm

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to 640x480
        frame = cv2.resize(frame, (640, 480))
        image = frame.copy()

        image = apply_cartoon_filter(frame)
            
        # Convert grayscale image back to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # cv2.imshow('Cartoon', image)
        # Write the processed frame
        out.write(image)

    # Release resources
    cap.release()
    out.release()

if __name__ == '__main__':
    input_path = "../data_collection/output/video/compiled_video.mp4"
    output_path = "../data_collection/output/video/compiled_video_edge.mp4"



    process_video(input_path, output_path)
    print(f"Processed video saved as {output_path}")

    print("All videos processed.")

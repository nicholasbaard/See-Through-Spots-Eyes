import cv2
import numpy as np
import random
from PIL import Image
from ultralytics import YOLO
import os

# Define the cartoon filter
def apply_cartoon_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)
    color = cv2.bilateralFilter(frame, 9, 150, 150)
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    def quantize_colors(image, k=16):
        data = image.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized_image = centers[labels.flatten()]
        quantized_image = quantized_image.reshape(image.shape)
        return quantized_image

    cartoon = quantize_colors(cartoon, k=5)
    cartoon = cv2.bitwise_and(cartoon, cartoon, mask=edges)
    cartoon = cv2.convertScaleAbs(cartoon, alpha=1.5, beta=0)
    return cartoon

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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to 640x480
        frame = cv2.resize(frame, (640, 480))
        image = frame.copy()

        image =  apply_cartoon_filter(frame)
        # Write the processed frame
        out.write(image)

    # Release resources
    cap.release()
    out.release()

if __name__ == '__main__':
    input_path = "../data_collection/output/video/compiled_video.mp4"
    output_path = "../data_collection/output/video/compiled_video_cartoon.mp4"



    process_video(input_path, output_path)
    print(f"Processed video saved as {output_path}")

    print("All videos processed.")

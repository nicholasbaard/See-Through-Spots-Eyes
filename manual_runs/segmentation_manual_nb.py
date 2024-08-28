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
        black_background = np.zeros_like(frame)

        # Perform segmentation
        results = model.predict(frame, conf=0.5, verbose=True)

        for result in results:
            if result.masks is not None:
                for i, mask in enumerate(result.masks.data):
                    # Convert mask to a binary numpy array
                    mask_binary = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask.numpy()
                    mask_binary = mask_binary > 0.5  # Thresholding to obtain binary mask

                    # Get the class ID for the mask
                    cls_id = int(result.boxes.cls[i])

                    # Create a tinted overlay using the class color
                    color_tint = np.full_like(frame, colors[cls_id], dtype=np.uint8)

                    # Resize the mask to match the frame dimensions
                    mask_binary_resized = cv2.resize(mask_binary.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                    # Apply the tinted overlay where the mask is True, keeping the background black
                    mask_binary_3ch = np.stack([mask_binary_resized] * 3, axis=-1)  # Convert mask to 3-channel
                    black_background = np.where(mask_binary_3ch, cv2.addWeighted(frame, 0.7, color_tint, 0.3, 0), black_background)

                    # Draw bounding boxes with class name and confidence
                    x1, y1, x2, y2 = map(int, result.boxes.xyxy[i])
                    confidence = result.boxes.conf[i]
                    label = f"{class_names[cls_id]} {confidence:.2f}"

                    # Draw the bounding box
                    cv2.rectangle(black_background, (x1, y1), (x2, y2), colors[cls_id], 2)

                    # Draw the label
                    cv2.putText(black_background, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls_id], 2)

        # Write the processed frame
        out.write(black_background)

    # Release resources
    cap.release()
    out.release()

if __name__ == '__main__':
    input_path = "../data_collection/output/video/compiled_video.mp4"
    output_path = "../data_collection/output/video/compiled_video_segmentation_nb.mp4"



    process_video(input_path, output_path)
    print(f"Processed video saved as {output_path}")

    print("All videos processed.")

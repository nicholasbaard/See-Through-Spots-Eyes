import cv2
import numpy as np
import random
from PIL import Image
from ultralytics import YOLO
import streamlit as st
import time

# Initialize YOLOv8 model for person detection
model = YOLO("../models/yolov8n-seg.pt")  # Use 'yolov8n-seg.pt' for segmentation
model.to("cpu")
class_names = model.names
print('Class Names:', class_names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

def webcam_preview(options: str, max_retries=5, retry_delay=1):
    stream_url = None #'rtsp://admin:admin@192.168.80.3:21554/media/video1'
    
    cap = None
    for attempt in range(max_retries):
        if stream_url is None:
            cap = cv2.VideoCapture(0)  # Use the first webcam
        else:
            cap = cv2.VideoCapture(stream_url)  # Use the provided stream URL

        if cap.isOpened():
            break
        else:
            st.warning(f"Failed to open video stream (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    else:
        st.error("Failed to open video stream after multiple attempts.")
        return

    # Set frame dimensions
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Get and set FPS
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print(f"Original FPS: {fps}")
    # cap.set(cv2.CAP_PROP_FPS, 5)  # Set the frame rate to 5 FPS
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print(f"Updated FPS: {fps}")
    frame_count = 0
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(retry_delay)
                ret, frame = cap.read()
                continue
            
                    # Resize frame to 640x640
            frame = cv2.resize(frame, (640, 480))
            black_background = np.zeros_like(frame)
            frame_count += 1
            if frame_count % 1 == 0:

                # Perform segmentation if the option is enabled
                if "Semantic Segmentation" in options:
                    results = model.predict(frame, conf=0.5, verbose=False)

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

                # Convert images to PIL format for further processing or display
                original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                filtered_image = Image.fromarray(cv2.cvtColor(black_background, cv2.COLOR_BGR2RGB))        
                yield original_image, filtered_image

        except Exception as e:
            # st.error(f"An error occurred: {e}. Retrying...")
            print(f"An error occurred: {e}. Retrying...")


if __name__ == '__main__':
    # Streamlit app layout
    st.set_page_config(layout="wide")

    # Load the header image
    header_image = Image.open("../streamlit_front_end/Header3_trim.png")

    # Display the header image
    st.image(header_image, use_column_width=True)
    
    # Dropdown menu for filter selection
    options = ["Semantic Segmentation"]

    # Display video stream with selected filter
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("What We See")
        original_placeholder = st.empty()
    with col2:
        st.header("What Spot Sees")
        filtered_placeholder = st.empty()
        
    footerimage = Image.open("../streamlit_front_end/Footer_trim.png")
    st.image(footerimage, use_column_width=True)
    
    for original_image, filtered_image in webcam_preview(options):
        if original_image is not None:
            original_placeholder.image(original_image, channels="RGB", use_column_width=True)
            filtered_placeholder.image(filtered_image, channels="RGB", use_column_width=True)

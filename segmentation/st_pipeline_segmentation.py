import cv2
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import random
import time
# Initialize YOLOv8 model for person detection
model = YOLO("../models/yolov8n-seg.pt")  # Use 'yolov8n-seg.pt' for segmentation
model.to("cpu")
class_names = model.names
print('Class Names: ', class_names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]


def webcam_preview(options:str, max_retries=5, retry_delay=1):
    
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

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Get the FPS of the capture
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    # cap.set(cv2.CAP_PROP_FPS, 5)  # Set the frame rate of the webcam
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print(f"FPS: {fps}")
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(retry_delay)
                ret, frame = cap.read()
                continue
            
            frame = cv2.resize(frame, (640, 480))
            image = frame.copy()

            #segmentation
            if "Semantic Segmentation" in options:
                results = model.predict(image, conf=0.5, verbose = False)
                image = results[0].plot()

            # cv2.imshow('frame', image)
            original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            filtered_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))        
            yield original_image, filtered_image
            
        except Exception as e:
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

    
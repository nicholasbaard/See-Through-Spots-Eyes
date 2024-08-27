import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io

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
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized_image = centers[labels.flatten()]
        quantized_image = quantized_image.reshape(image.shape)
        return quantized_image

    cartoon = quantize_colors(cartoon, k=16)
    cartoon = cv2.bitwise_and(cartoon, cartoon, mask=edges)
    cartoon = cv2.convertScaleAbs(cartoon, alpha=1.5, beta=0)
    return cartoon

# Function to capture video and yield frames
def video_stream():
    # Initialize webcam
    stream_url = None #'rtsp://admin:admin@192.168.6.168:21554/media/video1'

    if stream_url is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        st.error("Could not open webcam. Please check your webcam settings and permissions.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Get the FPS of the capture
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    cap.set(cv2.CAP_PROP_FPS, 5)  # Set the frame rate of the webcam
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            break

        original_frame = frame.copy()
        processed_frame = apply_cartoon_filter(frame)
        
        _, jpeg_original = cv2.imencode('.jpg', original_frame)
        _, jpeg_processed = cv2.imencode('.jpg', processed_frame)
        if ret:
            # Convert the frames to images for Streamlit
            original_image = Image.open(io.BytesIO(jpeg_original.tobytes()))
            processed_image = Image.open(io.BytesIO(jpeg_processed.tobytes()))
            yield original_image, processed_image

    cap.release()

if __name__ == "__main__":
    # Create a Streamlit app
    st.set_page_config(layout="wide")
    # Load and display the header image
    header_image = Image.open("../streamlit_front_end/Header3.png")
    st.image(header_image, use_column_width=True)

    # Create two columns for the videos
    col1, col2 = st.columns(2)

    # Display the original and processed video in separate columns
    with col1:
        st.header("What We See")
        original_placeholder = st.empty()

    with col2:
        st.header("What Spot Sees")
        filtered_placeholder = st.empty()
        
    # Load and display the footer image
    footer_image = Image.open("../streamlit_front_end/Footer.png")
    st.image(footer_image, use_column_width=True)

    # Stream video and apply cartoon filter
    for original_image, filtered_image in video_stream():
        if original_image is not None:
            original_placeholder.image(original_image, channels="RGB", use_column_width=True)
            filtered_placeholder.image(filtered_image, channels="RGB", use_column_width=True)



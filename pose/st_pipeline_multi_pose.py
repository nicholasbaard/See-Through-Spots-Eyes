import cv2
import mediapipe as mp
from ultralytics import YOLO
from PIL import Image
import streamlit as st
import time

# Initialize YOLOv8 model for person detection
model = YOLO("../models/yolov8n-seg.pt")  # Use 'yolov8n-seg.pt' for segmentation
model.to("cpu") 

# Initialize MediaPipe for pose estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.35)
main_landmarks = [
              mp_pose.PoseLandmark.NOSE,
              mp_pose.PoseLandmark.LEFT_EYE,
              mp_pose.PoseLandmark.RIGHT_EYE,
              mp_pose.PoseLandmark.LEFT_EAR,
              mp_pose.PoseLandmark.RIGHT_EAR,
              mp_pose.PoseLandmark.MOUTH_LEFT,
              mp_pose.PoseLandmark.MOUTH_RIGHT,
              mp_pose.PoseLandmark.LEFT_SHOULDER,
              mp_pose.PoseLandmark.RIGHT_SHOULDER,
              mp_pose.PoseLandmark.LEFT_ELBOW,
              mp_pose.PoseLandmark.RIGHT_ELBOW,
              mp_pose.PoseLandmark.LEFT_WRIST,
              mp_pose.PoseLandmark.RIGHT_WRIST,
              mp_pose.PoseLandmark.LEFT_HIP,
              mp_pose.PoseLandmark.RIGHT_HIP,
              mp_pose.PoseLandmark.LEFT_KNEE,
              mp_pose.PoseLandmark.RIGHT_KNEE,
              mp_pose.PoseLandmark.LEFT_ANKLE,
              mp_pose.PoseLandmark.RIGHT_ANKLE
          ]
# Draw connections between main landmarks
connections = [
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
            (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
            (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
            (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
            (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
        ]
# Generate a list of bright colors
bright_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
                (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128), (128, 0, 255), (0, 128, 255)]


def webcam_preview(max_retries=5, retry_delay=1):
    # Initialize webcam
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
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print(f"FPS: {fps}")
    # cap.set(cv2.CAP_PROP_FPS, 5)  # Set the frame rate of the webcam
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print(f"FPS: {fps}")
    frame_count = 0
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(retry_delay)
                ret, frame = cap.read()
                continue

            # Person detection
            frame = cv2.resize(frame, (640, 480))
            results = model.predict(frame, conf=0.5, verbose = False)

            image = frame.copy()
            frame_count += 1
            if frame_count % 1 == 0:
            # Iterate through detected objects
                for result in results:
                    if result.boxes is not None:
                        # Iterate through detected bounding boxes
                        for box in result.boxes:
                            # Only process if the detected object is a person
                            if result.names[int(box.cls[0])] == "person":
                                x1, y1, x2, y2 = map(int, box.xyxy[0])

                                # Extract the region of interest (ROI) for pose estimation
                                person_roi = image[y1:y2, x1:x2]

                                # Perform pose estimation on the detected person
                                pose_results = pose.process(person_roi)

                                # Annotate the original frame with pose landmarks
                                if pose_results.pose_landmarks:
                                    for landmark in main_landmarks:
                                        landmark_x = int(pose_results.pose_landmarks.landmark[landmark].x * (x2 - x1)) + x1
                                        landmark_y = int(pose_results.pose_landmarks.landmark[landmark].y * (y2 - y1)) + y1
                                        cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), -1)
                                        
                                    for i, connection in enumerate(connections):
                                        start_point = pose_results.pose_landmarks.landmark[connection[0]]
                                        end_point = pose_results.pose_landmarks.landmark[connection[1]]
                                        
                                        start_x, start_y = int(start_point.x * (x2 - x1)) + x1, int(start_point.y * (y2 - y1)) + y1
                                        end_x, end_y = int(end_point.x * (x2 - x1)) + x1, int(end_point.y * (y2 - y1)) + y1
                                        
                                        # Use a different bright color for each connection
                                        color = bright_colors[i % len(bright_colors)]
                                        cv2.line(image, (start_x, start_y), (end_x, end_y), color, 2)

                    
                original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                filtered_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                yield original_image, filtered_image

        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")

if __name__ == '__main__':
    
    # Streamlit app layout
    st.set_page_config(layout="wide")
    # Set the background image

    # Create a Streamlit app, verbose = False

    # Load the header image
    header_image = Image.open("../streamlit_front_end/Header3.png")

    # Display the header image
    st.image(header_image, use_column_width=True)
    
    # Display video stream with selected filter
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("What We See")
        original_placeholder = st.empty()
    with col2:
        st.header("What Spot Sees")
        filtered_placeholder = st.empty()
        
    footerimage = Image.open("../streamlit_front_end/Footer.png")
    st.image(footerimage, use_column_width=True)
    
    for original_image, filtered_image in webcam_preview():
        if original_image is not None:
            original_placeholder.image(original_image, channels="RGB", use_column_width=True)
            filtered_placeholder.image(filtered_image, channels="RGB", use_column_width=True)

    
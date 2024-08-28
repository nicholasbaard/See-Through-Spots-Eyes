import cv2
import numpy as np
import random
from PIL import Image
from ultralytics import YOLO
import os
import mediapipe as mp

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

        # Person detection
        results = model.predict(frame, conf=0.5, verbose=False)
        image = black_background.copy()
        
        # Iterate through detected objects
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Only process if the detected object is a person
                    if result.names[int(box.cls[0])] == "person":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Extract the region of interest (ROI) for pose estimation
                        person_roi = frame[y1:y2, x1:x2]

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

                    
        # Write the processed frame
        out.write(image)

    # Release resources
    cap.release()
    out.release()

if __name__ == '__main__':
    input_path = "../data_collection/output/video/compiled_video.mp4"
    output_path = "../data_collection/output/video/compiled_video_pose_nb.mp4"



    process_video(input_path, output_path)
    print(f"Processed video saved as {output_path}")

    print("All videos processed.")

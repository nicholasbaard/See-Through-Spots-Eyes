
import cv2
import os

def save_frames_from_video(video_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    frame_count = 0

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            break

        # Save the frame as an image file
        frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f"Saved {frame_count} frames to {output_folder}")

# Usage example
video_path = "output/video/output.avi"
output_folder = "output/frames"
save_frames_from_video(video_path, output_folder)

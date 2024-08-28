
import os
import cv2
from tqdm import tqdm
import re

def frames_to_video(input_folder, output_file, fps=25):
    # Get the list of frame files
    frame_files = [f for f in os.listdir(input_folder) if f.endswith('.png') or f.endswith('.jpg')]
    # frame_files.sort()  # Ensure frames are in order
    # Function to extract the number from the filename
    def extract_number(filename):
        match = re.search(r'\d+', filename)  # Extract digits from the filename
        return int(match.group()) if match else 0

    # Sort the list using the extracted number
    frame_files = sorted(frame_files, key=extract_number)
    
    if not frame_files:
        print("No frames found in the input folder.")
        return

    # Read the first frame to get dimensions
    first_frame = cv2.imread(os.path.join(input_folder, frame_files[0]))
    height, width, _ = first_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Write frames to video
    for frame_file in tqdm(frame_files, desc="Converting frames to video"):
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)
        out.write(frame)

    # Release the VideoWriter
    out.release()
    print(f"Video saved as {output_file}")

if __name__ == "__main__":
    input_folder = "output/frames"
    output_file = "output/video/compiled_video.mp4"
    fps = 25

    frames_to_video(input_folder, output_file, fps)

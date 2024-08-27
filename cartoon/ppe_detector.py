import cv2
import torch
import streamlit as st
from PIL import Image
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# Initialize Detectron2 predictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available (mps for Apple Silicon)
predictor = DefaultPredictor(cfg)

# Set the background image
# background_image = Image.open("background.jpg")

# Load the header image
header_image = Image.open("Header3.png")

# Display the header image
st.image(header_image)

# Video source selection
use_webcam = "Webcam"
uploaded_file = None

col1, col2 = st.columns(2)

# Create placeholders for the video streams
with col1:
    st.header("What We See")
    original_placeholder = st.empty()
with col2:
    st.header("What Spot Sees")
    detected_placeholder = st.empty()

# Display the footer image
footerimage = Image.open("Footer.png")
st.image(footerimage)

# Video processing logic
cap = cv2.VideoCapture(0) 

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display if using webcam
    frame = cv2.flip(frame, 1)

    # Run Detectron2 on the frame
    outputs = predictor(frame)

    # Create a blank frame to draw detected objects
    v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Convert visualized image to BGR
    detected_frame = v.get_image()[:, :, ::-1]

    # Display the original frame in the first column
    original_placeholder.image(frame, channels='BGR')

    # Display the frame with detected objects in the second column
    detected_placeholder.image(detected_frame, channels='BGR')

cap.release()



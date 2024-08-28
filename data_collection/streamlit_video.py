from PIL import Image
import streamlit as st

if __name__ == '__main__':
    # Streamlit app layout
    st.set_page_config(layout="wide")
    # Set the background image

    # Create a Streamlit app

    # Load the header image
    header_image = Image.open("../streamlit_front_end/Header3_trim.png")

    # Display the header image
    st.image(header_image, use_column_width=True)
    
    # Display video stream with selected filter
    col1, col2 = st.columns(2) 
    
    # Read the resized video
    # resized_video_file = open("output/video/output_h264.mp4", "rb")
    # video_bytes = resized_video_file.read()
    # video_file_proc = open("output/video/output_h264_seg.mp4", "rb")
    # video_bytes_proc = video_file_proc.read()
    
    with col1:
        st.header("What We See")
        st.video("output/video/output_h264.mp4", start_time=0, autoplay=True)
    with col2:
        st.header("What Spot Sees")
        st.video("output/video/output_h264_pencil.mp4", start_time=0, autoplay=True)
        
    footerimage = Image.open("../streamlit_front_end/Footer_trim.png")
    st.image(footerimage, use_column_width=True)
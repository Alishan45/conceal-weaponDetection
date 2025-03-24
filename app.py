import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # For Windows-specific issues

# Must be before any streamlit imports
import asyncio
import sys
if sys.platform == "win32":
    if sys.version_info >= (3, 8) and sys.version_info < (3, 9):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Now import other packages
import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile
import time

# Workaround for torch.classes error
import torch
if hasattr(torch.classes, '__path__'):
    torch.classes.__path__ = []  # Disable problematic path inspection

def main():
    # Set page config
    st.set_page_config(
        page_title="Thermal Pistol Detection",
        page_icon="🔫",
        layout="wide"
    )

    # Title and description
    st.title("Thermal Pistol Detection")
    st.markdown("""
        Upload an image/video or use webcam to detect thermal pistols.
        Model trained on custom thermal image dataset.
    """)

    # Sidebar options
    st.sidebar.header("Options")
    confidence = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.01)
    
    # Model loading with error handling
    @st.cache_resource
    def load_model():
        try:
            model = YOLO("best.pt")
            st.sidebar.success("Model loaded successfully!")
            return model
        except Exception as e:
            st.sidebar.error(f"Model failed to load: {str(e)}")
            return None

    model = load_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose media", type=["jpg", "png", "jpeg", "mp4"])

    # Webcam option
    use_webcam = st.sidebar.checkbox("Use Webcam")
    if use_webcam:
        st.warning("Webcam access requires browser permission")
        run_webcam = st.checkbox("Start Webcam Feed")
        
        if run_webcam:
            FRAME_WINDOW = st.image([])
            cap = cv2.VideoCapture(0)
            
            while run_webcam and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Convert and display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)
                
                # Add small delay
                time.sleep(0.1)
            
            cap.release()

    # Media processing
    if uploaded_file is not None and model is not None:
        if uploaded_file.type.startswith("image"):
            # Image processing
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)
            
            with st.spinner("Detecting..."):
                results = model(img_array, conf=confidence)
                plotted = results[0].plot()
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", use_column_width=True)
            with col2:
                st.image(plotted, caption="Detected", use_column_width=True)
        
        elif uploaded_file.type.startswith("video"):
            # Video processing
            st.warning("Video processing may take time...")
            
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame, conf=confidence)
                plotted = results[0].plot()
                stframe.image(plotted)
            
            cap.release()
            os.unlink(tfile.name)

if __name__ == "__main__":
    main()

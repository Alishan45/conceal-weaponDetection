import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile
import os

# Set page title and icon
st.set_page_config(
    page_title="Thermal Pistol Detection",
    page_icon="🔫",
    layout="wide"
)

# Title and description
st.title("Thermal Pistol Detection using YOLOv11")
st.markdown("""
    Upload an image or video, or use your webcam to detect thermal pistols in real-time.
    The model is trained on a custom dataset of thermal images.
""")

# Sidebar for additional options
st.sidebar.header("Options")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
class_names = st.sidebar.multiselect("Filter Classes", ["pistol"], default=["pistol"])

# Map class names to class IDs
class_name_to_id = {"pistol": 0}  # Update this based on your dataset
class_filter = [class_name_to_id[cls] for cls in class_names]

# Dark/Light mode toggle
dark_mode = st.sidebar.checkbox("Dark Mode", value=True)
if dark_mode:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Load the YOLOv11 model
@st.cache_resource
def load_model():
    return YOLO("/content/runs/detect/yolo11n_finetuned/weights/best.pt")

model = load_model()

# Upload image or video
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

# Real-time webcam inference
st.sidebar.header("Real-Time Webcam")
use_webcam = st.sidebar.checkbox("Use Webcam")

# About section
st.sidebar.header("About")
st.sidebar.markdown("""
    This app uses a fine-tuned YOLOv11 model to detect thermal pistols in images and videos.
    - **Model**: YOLOv11n
    - **Dataset**: Custom thermal pistol dataset
    - **Confidence Threshold**: Adjustable
    - **Class Filtering**: Select specific classes to detect
""")

if use_webcam:
    st.subheader("Real-Time Webcam Inference")
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture video from webcam.")
            break

        # Run inference with class filtering (if classes are selected)
        if class_filter:
            results = model(frame, conf=confidence_threshold, classes=class_filter)
        else:
            results = model(frame, conf=confidence_threshold)

        # Display the results
        plotted_image = results[0].plot(line_width=2)
        FRAME_WINDOW.image(plotted_image, channels="BGR")

    camera.release()
else:
    if uploaded_file is not None:
        # Check if the file is an image or video
        if uploaded_file.type.startswith("image"):
            # Read the image
            image = Image.open(uploaded_file)
            image = np.array(image)  # Convert to numpy array

            # Display the original image
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

            # Run inference with class filtering (if classes are selected)
            with st.spinner("Running inference..."):
                if class_filter:
                    results = model(image, conf=confidence_threshold, classes=class_filter)
                else:
                    results = model(image, conf=confidence_threshold)

            # Display the results
            st.subheader("Detected Pistols")
            plotted_image = results[0].plot(line_width=2)
            st.image(plotted_image, use_container_width=True, caption="Detected Pistols")

            # Display detection details
            st.write("Detection Details:")
            for box in results[0].boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                bbox = box.xyxy[0].tolist()  # Bounding box coordinates
                st.write(f"""
                    - **Class ID**: {class_id}
                    - **Confidence**: {confidence:.2f}
                    - **Bounding Box**: {bbox}
                """)

            # Download the results
            result_image = Image.fromarray(plotted_image)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                result_image.save(tmpfile.name)
                with open(tmpfile.name, "rb") as file:
                    st.download_button(
                        label="Download Result",
                        data=file,
                        file_name="detected_pistols.jpg",
                        mime="image/jpeg"
                    )

        elif uploaded_file.type.startswith("video"):
            # Save the uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                tmpfile.write(uploaded_file.read())
                video_path = tmpfile.name

            # Display the original video
            st.subheader("Original Video")
            st.video(video_path)

            # Run inference on the video
            with st.spinner("Running inference on video..."):
                cap = cv2.VideoCapture(video_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                output_path = "output_video.mp4"
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Run inference with class filtering (if classes are selected)
                    if class_filter:
                        results = model(frame, conf=confidence_threshold, classes=class_filter)
                    else:
                        results = model(frame, conf=confidence_threshold)

                    # Plot the results
                    plotted_frame = results[0].plot(line_width=2)
                    out.write(plotted_frame)

                cap.release()
                out.release()

            # Display the results
            st.subheader("Detected Pistols in Video")
            st.video(output_path)

            # Download the results
            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download Result Video",
                    data=file,
                    file_name="detected_pistols.mp4",
                    mime="video/mp4"
                )

            # Clean up temporary files
            os.remove(video_path)
            os.remove(output_path)
    else:
        st.info("Please upload an image or video to get started.")

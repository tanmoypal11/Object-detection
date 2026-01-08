import PIL
import streamlit as st
from pathlib import Path
import os

# Local Modules
import settings
import helper

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Image Detection | SmartVision AI",
    page_icon="🤖",
    layout="wide"
)

# Main page heading
st.title("🎯 YOLOv8 Image Detection")
st.write("Upload an image to detect or segment objects using YOLOv8.")

# -------------------------------------------------
# SIDEBAR - MODEL CONFIGURATION
# -------------------------------------------------
st.sidebar.header("ML Model Config")
model_type = st.sidebar.radio("Select Task", ['Detection', 'Segmentation'])
confidence = float(st.sidebar.slider("Model Confidence", 25, 100, 45)) / 100

# Path selection
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
else:
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check path: {model_path}")
    st.error(ex)

# -------------------------------------------------
# IMAGE UPLOAD LOGIC
# -------------------------------------------------
st.sidebar.header("Image Upload")
source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

col1, col2 = st.columns(2)

# --- COLUMN 1: SOURCE IMAGE ---
with col1:
    st.subheader("📷 Source Image")
    if source_img is None:
        # Check if default image exists before opening
        if os.path.exists(settings.DEFAULT_IMAGE):
            default_image = PIL.Image.open(settings.DEFAULT_IMAGE)
            st.image(default_image, caption="Default Image", use_container_width=True)
        else:
            st.warning("No default image found. Please upload an image in the sidebar.")
    else:
        uploaded_image = PIL.Image.open(source_img)
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

# --- COLUMN 2: RESULT IMAGE ---
with col2:
    st.subheader("🔍 Result Image")
    if source_img is None:
        # Check if default result image exists
        if os.path.exists(settings.DEFAULT_DETECT_IMAGE):
            default_detected_image = PIL.Image.open(settings.DEFAULT_DETECT_IMAGE)
            st.image(default_detected_image, caption='Default Result', use_container_width=True)
        else:
            st.info("Detection results will appear here after you upload an image.")
    else:
        # This button only appears when an image is uploaded
        if st.sidebar.button('Detect Objects'):
            with st.spinner("Analyzing..."):
                # YOLO Inference
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                
                # Plot results (BGR to RGB conversion)
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Objects', use_container_width=True)
                
                # Metadata Expander
                try:
                    with st.expander("Detection Metadata"):
                        if len(boxes) > 0:
                            for box in boxes:
                                st.write(box.data)
                        else:
                            st.write("No objects detected at this confidence level.")
                except Exception:
                    st.write("Error displaying metadata.")
import PIL
import streamlit as st
from pathlib import Path

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Image Detection | SmartVision AI",
    page_icon="🤖",
    layout="wide"
)

# Main page heading
st.title("🎯 YOLOv8 Image Detection")
st.write("Upload an image to detect or segment objects using YOLOv8.")

# Sidebar - Model Configuration
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

# Image Upload Logic
st.sidebar.header("Image Upload")
source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Source Image")
    if source_img is None:
        default_image = PIL.Image.open(settings.DEFAULT_IMAGE)
        st.image(default_image, caption="Default Image", use_container_width=True)
    else:
        uploaded_image = PIL.Image.open(source_img)
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("🔍 Result Image")
    if source_img is None:
        default_detected_image = PIL.Image.open(settings.DEFAULT_DETECT_IMAGE)
        st.image(default_detected_image, caption='Default Result', use_container_width=True)
    else:
        if st.sidebar.button('Detect Objects'):
            with st.spinner("Analyzing..."):
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                # Plot and convert BGR to RGB
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Objects', use_container_width=True)
                
                try:
                    with st.expander("Detection Metadata"):
                        for box in boxes:
                            st.write(box.data)
                except Exception:
                    st.write("No objects detected!")
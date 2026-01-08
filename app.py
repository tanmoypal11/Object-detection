import PIL
import streamlit as st
from pathlib import Path
import os

# Local Modules
import settings
import helper

# 1. PAGE CONFIG
st.set_page_config(
    page_title="Image Detection | SmartVision AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🎯 YOLOv8 Image Detection")
st.write("Upload an image to detect or segment objects.")

# 2. MODEL CONFIG
st.sidebar.header("ML Model Config")
model_type = st.sidebar.radio("Select Task", ['Detection', 'Segmentation'])
confidence = float(st.sidebar.slider("Model Confidence", 25, 100, 45)) / 100

if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
else:
    model_path = Path(settings.SEGMENTATION_MODEL)

try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check path: {model_path}")
    st.stop() # Stop the app here if model fails

# 3. IMAGE UPLOAD
st.sidebar.header("Image Upload")
source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

col1, col2 = st.columns(2)

# --- LEFT COLUMN: SOURCE ---
with col1:
    st.subheader("📷 Source Image")
    if source_img is None:
        # SAFE CHECK: Only open if file exists
        if os.path.exists(str(settings.DEFAULT_IMAGE)):
            default_image = PIL.Image.open(str(settings.DEFAULT_IMAGE))
            st.image(default_image, caption="Default Image", use_container_width=True)
        else:
            st.warning("Default placeholder image not found on server.")
    else:
        uploaded_image = PIL.Image.open(source_img)
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

# --- RIGHT COLUMN: RESULT ---
with col2:
    st.subheader("🔍 Result Image")
    if source_img is None:
        # SAFE CHECK: Only open if file exists
        if os.path.exists(str(settings.DEFAULT_DETECT_IMAGE)):
            default_detected_image = PIL.Image.open(str(settings.DEFAULT_DETECT_IMAGE))
            st.image(default_detected_image, caption='Default Result', use_container_width=True)
        else:
            st.info("Upload an image to see detection results.")
    else:
        if st.sidebar.button('Detect Objects'):
            with st.spinner("Analyzing..."):
                res = model.predict(uploaded_image, conf=confidence)
                # Plot and convert BGR to RGB
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Objects', use_container_width=True)
                
                with st.expander("Detection Metadata"):
                    if res[0].boxes:
                        for box in res[0].boxes:
                            st.write(box.data)
                    else:
                        st.write("No objects detected.")
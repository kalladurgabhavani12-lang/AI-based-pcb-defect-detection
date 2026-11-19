import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="AI-Based PCB Defect Detection (YOLOv8)", layout="wide")

DEFECT_INFO = {
    "missing_hole": "A required drill hole/via is absent on the PCB substrate.",
    "mouse_bite": "Small indentations along the edge of a copper trace caused by improper depaneling.",
    "open_circuit": "A break or gap in a conductive copper trace preventing electrical continuity.",
    "short": "Unintended copper bridge connecting two adjacent conductive tracks.",
    "spur": "Unwanted sharp copper protrusion extending from a trace.",
    "spurious_copper": "Isolated patch of excess copper left behind during the etching process."
}

@st.cache_resource
def load_yolo_model():
    return YOLO("best_pcb_yolo.pt")

st.title("🔍 AI-Based PCB Defect Detection System (YOLOv8)")
st.markdown("**Real-Time Automated Visual Inspection & Defect Localization using YOLOv8**")
st.divider()

st.sidebar.header("Project Details")
st.sidebar.write("**Department:** Electronics & Communication Engineering")
st.sidebar.write("**Institute:** Dadi Institute of Engineering & Technology")
st.sidebar.write("**Model Architecture:** YOLOv8 (Object Detection)")
conf_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.10, 0.90, 0.25, 0.05)

try:
    model = load_yolo_model()
except Exception:
    st.error("Model file `best_pcb_yolo.pt` not found! Train your YOLOv8 model first and upload `best_pcb_yolo.pt` to GitHub.")
    st.stop()

uploaded_file = st.file_uploader("Upload a PCB Image for Defect Detection (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # Noise removal preprocessing before YOLOv8 inference
    denoised_img = cv2.GaussianBlur(img_np, (3, 3), 0)

    start_time = time.time()
    results = model.predict(source=denoised_img, conf=conf_threshold)
    inference_ms = (time.time() - start_time) * 1000

    annotated_bgr = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original PCB Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("YOLOv8 Detected Defects (Bounding Boxes)")
        st.image(annotated_rgb, use_container_width=True)

    st.divider()
    boxes = results[0].boxes
    m1, m2 = st.columns(2)
    m1.metric("Total Defects Localized", len(boxes))
    m2.metric("Inference Speed", f"{inference_ms:.1f} ms")

    if len(boxes) == 0:
        st.success("No defects detected above the selected confidence threshold.")
    else:
        st.subheader("Detected Defect Breakdown")
        for idx, box in enumerate(boxes, start=1):
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0]) * 100
            desc = DEFECT_INFO.get(cls_name.lower(), "Manufacturing defect detected on PCB trace.")
            st.warning(f"**#{idx} — {cls_name.replace('_', ' ').title()}** ({conf:.1f}% confidence): {desc}")

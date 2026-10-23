
import io
import threading
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import (
    VideoProcessorBase,
    webrtc_streamer,
)

# ---------------- CONFIGURATION ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"

st.set_page_config(
    page_title="YOLO11 Weapon Detection",
    page_icon="🔍",
    layout="wide",
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model(path):
    from ultralytics import YOLO
    return YOLO(str(path))

if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Unable to load YOLO model: {exc}")
    st.stop()

MODEL_LOCK = threading.Lock()


def detect_image(image_rgb, confidence):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    with MODEL_LOCK:
        result = model.predict(
            source=image_bgr,
            conf=confidence,
            imgsz=640,
            verbose=False,
        )[0]

    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(
        annotated_bgr, cv2.COLOR_BGR2RGB
    )
    detections = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        detections.append({
            "Class": result.names[class_id],
            "Confidence": round(float(box.conf.item()), 4),
        })

    return annotated_rgb, detections


# ---------------- WEBCAM PROCESSOR ----------------
class WeaponProcessor(VideoProcessorBase):
    def __init__(self):
        self.confidence = 0.35

    def recv(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")

        try:
            with MODEL_LOCK:
                result = model.predict(
                    source=frame_bgr,
                    conf=self.confidence,
                    imgsz=640,
                    verbose=False,
                )[0]

            annotated = result.plot()
        except Exception:
            annotated = frame_bgr

        return av.VideoFrame.from_ndarray(
            annotated, format="bgr24"
        )


# ---------------- WEBSITE ----------------
st.title("YOLO11 Weapon Detection")
st.write(
    "Detect knives and pistols using a live webcam "
    "or an uploaded image. No camera is needed for image testing."
)
st.warning(
    "AI detections can be incorrect. A missing detection "
    "does not prove that a weapon is absent."
)

confidence = st.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.35,
    step=0.05,
)

camera_tab, image_tab = st.tabs(
    ["Live Webcam", "Upload Image"]
)

# ---------------- LIVE WEBCAM ----------------
with camera_tab:
    st.subheader("Live camera detection")
    st.info(
        "Allow camera permission in your browser, then click START. "
        "Live camera access works from the deployed HTTPS website "
        "or localhost."
    )

    ctx = webrtc_streamer(
        key="weapon-camera",
        video_processor_factory=WeaponProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.confidence = confidence

# ---------------- IMAGE UPLOAD ----------------
with image_tab:
    uploaded = st.file_uploader(
        "Upload a JPG, JPEG, PNG, or WEBP image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded image",
                     use_container_width=True)

            if st.button("Detect weapons", type="primary"):
                with st.spinner("Running YOLO detection..."):
                    annotated, detections = detect_image(
                        np.array(image), confidence
                    )

                st.subheader("Detection result")
                st.image(annotated, use_container_width=True)

                if detections:
                    st.success(
                        f"Detected {len(detections)} object(s)."
                    )
                    st.dataframe(
                        detections, use_container_width=True
                    )
                else:
                    st.info(
                        "No objects detected above the selected "
                        "confidence threshold."
                    )

                buffer = io.BytesIO()
                Image.fromarray(annotated).save(
                    buffer, format="PNG"
                )

                st.download_button(
                    "Download annotated image",
                    data=buffer.getvalue(),
                    file_name="weapon_detection_result.png",
                    mime="image/png",
                )

        except Exception as exc:
            st.error(f"Image processing failed: {exc}")

st.caption(
    "For demonstration purposes. Verify predictions independently."
)

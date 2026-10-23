
import io
import time
import threading
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from ultralytics import YOLO


# =====================================================
# CONFIGURATION
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"
IMAGE_SIZE = 416
MAX_IMAGE_SIDE = 960

st.set_page_config(
    page_title="YOLO11 Weapon Detection",
    page_icon="🔍",
    layout="wide",
)

st.title("YOLO11 Weapon Detection")
st.write(
    "Detect knives and pistols using a webcam or an uploaded image."
)
st.caption(
    "AI predictions may be incorrect. Verify results independently."
)


# =====================================================
# LOAD MODEL ONCE
# =====================================================
@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    with st.spinner("Loading YOLO model for the first time..."):
        model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Model loading failed: {type(exc).__name__}: {exc}")
    st.stop()

MODEL_LOCK = threading.Lock()


# =====================================================
# IMAGE DETECTION
# =====================================================
def detect_image(image, confidence):
    # Resize large images while preserving aspect ratio
    image = image.convert("RGB")
    image.thumbnail(
        (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE),
        Image.Resampling.LANCZOS,
    )

    image_rgb = np.asarray(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    start_time = time.perf_counter()

    with MODEL_LOCK:
        result = model.predict(
            source=image_bgr,
            conf=confidence,
            imgsz=IMAGE_SIZE,
            device="cpu",
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

    elapsed = time.perf_counter() - start_time
    return annotated_rgb, detections, elapsed


# =====================================================
# WEBCAM PROCESSOR
# =====================================================
class WeaponProcessor(VideoProcessorBase):
    def __init__(self):
        self.confidence = 0.35

    def recv(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")

        # Reduce webcam resolution for faster CPU inference
        height, width = frame_bgr.shape[:2]
        max_width = 640

        if width > max_width:
            new_height = int(height * max_width / width)
            frame_bgr = cv2.resize(
                frame_bgr,
                (max_width, new_height),
                interpolation=cv2.INTER_AREA,
            )

        try:
            with MODEL_LOCK:
                result = model.predict(
                    source=frame_bgr,
                    conf=self.confidence,
                    imgsz=IMAGE_SIZE,
                    device="cpu",
                    verbose=False,
                )[0]

            annotated = result.plot()

        except Exception:
            annotated = frame_bgr

        return av.VideoFrame.from_ndarray(
            annotated, format="bgr24"
        )


# =====================================================
# USER SETTINGS
# =====================================================
confidence = st.slider(
    "Detection confidence",
    min_value=0.10,
    max_value=0.95,
    value=0.35,
    step=0.05,
)

camera_tab, image_tab = st.tabs(
    ["Live Webcam", "Upload Image"]
)


# =====================================================
# LIVE WEBCAM
# =====================================================
with camera_tab:
    st.subheader("Live camera detection")

    st.info(
        "Click START and allow camera access in your browser. "
        "Live camera requires HTTPS when accessed remotely."
    )

    ctx = webrtc_streamer(
        key="weapon-camera",
        video_processor_factory=WeaponProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        },
        async_processing=True,
    )

    if ctx.video_processor is not None:
        ctx.video_processor.confidence = confidence


# =====================================================
# IMAGE UPLOAD
# =====================================================
with image_tab:
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="weapon-image",
    )

    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")

            st.image(
                image,
                caption="Uploaded image",
                use_container_width=True,
            )

            if st.button("Detect weapons", type="primary"):
                try:
                    with st.spinner(
                        "Running YOLO detection..."
                    ):
                        annotated, detections, elapsed = detect_image(
                            image.copy(), confidence
                        )

                    st.success(
                        f"Detection completed in {elapsed:.2f} seconds."
                    )

                    st.subheader("Detection result")
                    st.image(
                        annotated,
                        use_container_width=True,
                    )

                    if detections:
                        st.success(
                            f"Detected {len(detections)} object(s)."
                        )
                        st.dataframe(
                            detections,
                            use_container_width=True,
                        )
                    else:
                        st.info(
                            "No objects detected above this confidence "
                            "threshold. This does not prove that a "
                            "weapon is absent."
                        )

                    output = io.BytesIO()
                    Image.fromarray(annotated).save(
                        output, format="PNG"
                    )

                    st.download_button(
                        "Download detection result",
                        data=output.getvalue(),
                        file_name="weapon_detection_result.png",
                        mime="image/png",
                    )

                except Exception as exc:
                    st.error(
                        f"Detection failed: {type(exc).__name__}: {exc}"
                    )

        except Exception as exc:
            st.error(
                f"Image loading failed: {type(exc).__name__}: {exc}"
            )

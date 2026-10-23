
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
MAX_IMAGE_SIDE = 640
DEFAULT_CONFIDENCE = 0.25

st.set_page_config(
    page_title="YOLO11 Weapon Detection",
    page_icon="🔍",
    layout="wide",
)


# =====================================================
# LOAD MODEL ONCE
# =====================================================
@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


st.title("🔍 YOLO11 Weapon Detection")
st.write(
    "Upload an image for automatic weapon detection, "
    "or use your webcam when available."
)

if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    with st.spinner("Loading weapon detection model..."):
        model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Model loading failed: {type(exc).__name__}: {exc}")
    st.stop()

MODEL_LOCK = threading.Lock()


# =====================================================
# DETECTION FUNCTION
# =====================================================
def detect_weapons(image, confidence):
    # Resize large images to reduce CPU processing time.
    image = image.convert("RGB")
    image.thumbnail(
        (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE),
        Image.Resampling.LANCZOS,
    )

    image_rgb = np.asarray(image)
    image_bgr = cv2.cvtColor(
        image_rgb, cv2.COLOR_RGB2BGR
    )

    start_time = time.perf_counter()

    with MODEL_LOCK:
        result = model.predict(
            source=image_bgr,
            conf=confidence,
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False,
        )[0]

    elapsed = time.perf_counter() - start_time

    # Generate the annotated image.
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(
        annotated_bgr, cv2.COLOR_BGR2RGB
    )

    detections = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        score = float(box.conf.item())
        xyxy = box.xyxy[0].tolist()

        detections.append({
            "Class": result.names[class_id],
            "Confidence": round(score, 4),
            "X1": round(xyxy[0], 1),
            "Y1": round(xyxy[1], 1),
            "X2": round(xyxy[2], 1),
            "Y2": round(xyxy[3], 1),
        })

    return annotated_rgb, detections, elapsed


# =====================================================
# WEBCAM PROCESSOR
# =====================================================
class WeaponProcessor(VideoProcessorBase):
    def __init__(self):
        self.confidence = DEFAULT_CONFIDENCE

    def recv(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")

        # Reduce frame size for CPU inference.
        height, width = frame_bgr.shape[:2]

        if width > 480:
            new_height = int(height * 480 / width)
            frame_bgr = cv2.resize(
                frame_bgr,
                (480, new_height),
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
            # Keep the camera stream alive if a frame fails.
            annotated = frame_bgr

        return av.VideoFrame.from_ndarray(
            annotated, format="bgr24"
        )


# =====================================================
# SETTINGS
# =====================================================
confidence = st.slider(
    "Detection confidence threshold",
    min_value=0.10,
    max_value=0.95,
    value=DEFAULT_CONFIDENCE,
    step=0.05,
    help="Lower values may detect more objects but can increase false positives.",
)

camera_tab, image_tab = st.tabs(
    ["📷 Live Webcam", "🖼️ Automatic Image Detection"]
)


# =====================================================
# LIVE WEBCAM
# =====================================================
with camera_tab:
    st.subheader("Live Weapon Detection")

    st.info(
        "Click START to enable your browser camera. "
        "Allow camera permission when prompted. "
        "This feature requires a camera on the device "
        "opening the website."
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
# AUTOMATIC IMAGE DETECTION
# =====================================================

# =====================================================
# AUTOMATIC IMAGE DETECTION
# =====================================================
with image_tab:
    uploaded = st.file_uploader(
        "Upload an image for automatic weapon detection",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="weapon-image-upload",
    )

    if uploaded is not None:
        try:
            image_bytes = uploaded.getvalue()
            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            st.image(
                image,
                caption="Uploaded image",
                use_container_width=True,
            )

            # Run inference only once per uploaded image
            # and confidence setting during this session.
            if (
                st.session_state.get("last_image_id")
                != (uploaded.name, len(image_bytes), confidence)
            ):
                st.session_state.pop("detection_result", None)
                st.session_state["last_image_id"] = (
                    uploaded.name, len(image_bytes), confidence
                )

                with st.spinner("Detecting weapons..."):
                    annotated, detections, elapsed = detect_weapons(
                        image.copy(), confidence
                    )

                buffer = io.BytesIO()
                Image.fromarray(annotated).save(
                    buffer, format="PNG"
                )

                st.session_state["detection_result"] = {
                    "image": buffer.getvalue(),
                    "detections": detections,
                    "elapsed": elapsed,
                }

            output = st.session_state.get("detection_result")

            if output is not None:
                st.success(
                    f"Detection completed in "
                    f"{output['elapsed']:.2f} seconds."
                )

                st.subheader("Detection Result")
                st.image(
                    output["image"],
                    use_container_width=True,
                )

                if output["detections"]:
                    st.dataframe(
                        output["detections"],
                        use_container_width=True,
                    )
                else:
                    st.info(
                        "No objects detected at this confidence "
                        "threshold."
                    )

                st.download_button(
                    "Download detection result",
                    data=output["image"],
                    file_name="weapon_detection_result.png",
                    mime="image/png",
                )

        except Exception as exc:
            st.error(
                f"Detection failed: {type(exc).__name__}: {exc}"
            )


# =====================================================
# FOOTER
# =====================================================
st.caption(
    "YOLO predictions are estimates. Always verify detections "
    "independently before making decisions."
)

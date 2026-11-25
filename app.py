
import streamlit as st
import tempfile
import threading
import time
import numpy as np
import av

from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="Weapon Detection System",
    page_icon="🛡️",
    layout="wide"
)

API_KEY = st.secrets["ROBOFLOW_API_KEY"]
WORKSPACE = st.secrets["ROBOFLOW_WORKSPACE"]
WORKFLOW_ID = st.secrets["ROBOFLOW_WORKFLOW_ID"]

CONFIDENCE_THRESHOLD = 0.30
LIVE_INTERVAL = 1.5  # seconds between live inference requests

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

# ---------------- PREDICTION FUNCTION ----------------

def get_predictions(image):
    """Send an image to the Roboflow Workflow."""

    with tempfile.NamedTemporaryFile(
        suffix=".jpg", delete=False
    ) as temp:
        image.convert("RGB").save(temp.name, format="JPEG")
        image_path = temp.name

    try:
        result = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={"image": image_path}
        )
    finally:
        import os
        if os.path.exists(image_path):
            os.remove(image_path)

    if isinstance(result, list):
        result = result[0] if result else {}

    predictions = result.get("predictions", [])

    if isinstance(predictions, dict):
        predictions = predictions.get("predictions", [])

    return [
        p for p in predictions
        if float(p.get("confidence", 0)) >= CONFIDENCE_THRESHOLD
    ]


# ---------------- DRAW DETECTION BOXES ----------------

def draw_boxes(image, predictions):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    for p in predictions:
        x = float(p.get("x", 0))
        y = float(p.get("y", 0))
        w = float(p.get("width", 0))
        h = float(p.get("height", 0))

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        label = p.get("class", "Weapon")
        confidence = float(p.get("confidence", 0))

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=4
        )

        draw.text(
            (x1, max(0, y1 - 20)),
            f"{label}: {confidence:.1%}",
            fill="red"
        )

    return image


# ---------------- LIVE CAMERA PROCESSOR ----------------

class WeaponProcessor(VideoProcessorBase):

    def __init__(self):
        self.last_inference = 0
        self.predictions = []
        self.lock = threading.Lock()
        self.error = None

    def recv(self, frame):

        image_array = frame.to_ndarray(format="rgb24")
        image = Image.fromarray(image_array)

        current_time = time.monotonic()

        if current_time - self.last_inference >= LIVE_INTERVAL:

            self.last_inference = current_time

            try:
                predictions = get_predictions(image)

                with self.lock:
                    self.predictions = predictions
                    self.error = None

            except Exception as e:
                with self.lock:
                    self.error = str(e)

        with self.lock:
            predictions = list(self.predictions)

        output_image = draw_boxes(image, predictions)

        return av.VideoFrame.from_ndarray(
            np.array(output_image),
            format="rgb24"
        )


# ---------------- STREAMLIT UI ----------------

st.title("🛡️ Weapon Detection System")
st.write("Detect weapons using your live camera or an uploaded image.")

camera_tab, upload_tab = st.tabs([
    "📷 Live Camera Detection",
    "🖼️ Image Upload Detection"
])

# ---------------- CAMERA TAB ----------------

with camera_tab:

    st.subheader("Live Weapon Detection")

    st.info(
        "Click START to activate your camera. "
        "Allow camera permission when your browser asks."
    )

    webrtc_streamer(
        key="weapon-live-camera",
        video_processor_factory=WeaponProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    st.caption(
        "Live inference is performed approximately every "
        "1.5 seconds. Actual speed depends on network and API response time."
    )


# ---------------- IMAGE UPLOAD TAB ----------------

with upload_tab:

    st.subheader("Upload an Image")

    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG or PNG image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("Detect Weapons", type="primary"):

            with st.spinner("Detecting weapons..."):

                try:
                    predictions = get_predictions(image)
                    result_image = draw_boxes(image, predictions)

                    st.success("Detection completed!")

                    st.image(
                        result_image,
                        caption="Detection Result",
                        use_container_width=True
                    )

                    st.metric(
                        "Detected Objects",
                        len(predictions)
                    )

                    if predictions:
                        st.write("### Detection Details")

                        for p in predictions:
                            st.write(
                                f"- {p.get('class', 'Object')} — "
                                f"{float(p.get('confidence', 0)):.2%}"
                            )
                    else:
                        st.info("No objects detected above the confidence threshold.")

                    st.download_button(
                        "Download Detection Result",
                        data=image_to_bytes(result_image),
                        file_name="weapon_detection_result.jpg",
                        mime="image/jpeg"
                    )

                except Exception as e:
                    st.error(f"Detection failed: {e}")


def image_to_bytes(image):
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()

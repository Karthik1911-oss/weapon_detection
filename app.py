
import os
import time
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from inference_sdk import (
    InferenceHTTPClient,
    InferenceConfiguration
)


# =========================================================
# 1. STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Weapon Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. ROBOFLOW CONFIGURATION
# Exact configuration from your Kaggle notebook
# =========================================================

WORKSPACE = "karthiks-workspace-0hwsm"

WORKFLOW = (
    "weapon-detection-gqi8p-1-rfdetr-small-t1-"
    "vweapon-detection-gqi8p-1-rfdetr-small-t1-logic"
)

# Read API key from Streamlit Cloud Secrets or environment
try:
    API_KEY = st.secrets.get("ROBOFLOW_API_KEY", "")
except Exception:
    API_KEY = ""

if not API_KEY:
    API_KEY = os.getenv("ROBOFLOW_API_KEY", "")


# =========================================================
# 3. PAGE STYLING
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #10141d;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.hero {
    background: linear-gradient(120deg, #18263d, #293b55);
    padding: 28px;
    border-radius: 15px;
    margin-bottom: 25px;
    border: 1px solid #43546d;
}

.hero h1 {
    color: white;
    margin-bottom: 8px;
}

.hero p {
    color: #d5deeb;
    font-size: 16px;
}

.metric-card {
    background: #202b3a;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #35445a;
    text-align: center;
}

.metric-value {
    font-size: 30px;
    font-weight: bold;
    color: white;
}

.metric-label {
    color: #bdc9d9;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. HEADER
# =========================================================

st.markdown("""
<div class="hero">

<h1>🔍 AI Weapon Detection System</h1>

<p>
RF-DETR Small powered weapon detection using
camera capture and image upload.
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# 5. CHECK API KEY
# =========================================================

if not API_KEY:

    st.error(
        "Roboflow API key is missing. "
        "Configure ROBOFLOW_API_KEY in Streamlit Cloud Secrets."
    )

    st.stop()


# =========================================================
# 6. INITIALIZE ROBOFLOW CLIENT
# =========================================================

@st.cache_resource
def get_client(api_key):

    inference_client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key
    )

    inference_client.configure(
        InferenceConfiguration(
            api_key_transport="header"
        )
    )

    return inference_client


client = get_client(API_KEY)


# =========================================================
# 7. DETECTION FUNCTION
# Uses the exact response structure from your notebook
# =========================================================

def detect_weapons(image_path, min_conf=0.30, retries=2):

    raw = None

    for attempt in range(retries + 1):

        try:

            raw = client.run_workflow(
                workspace_name=WORKSPACE,
                workflow_id=WORKFLOW,
                images={
                    "image": image_path
                }
            )

            break

        except Exception as error:

            if attempt == retries:

                raise RuntimeError(
                    f"Roboflow workflow failed: {error}"
                ) from error

            time.sleep(2 ** attempt)

    if not isinstance(raw, list) or not raw:
        raise RuntimeError(
            "Roboflow returned an empty or invalid response."
        )

    block = raw[0].get("predictions")

    if not isinstance(block, dict):
        raise RuntimeError(
            "The workflow response does not contain "
            "the expected predictions object."
        )

    predictions = block.get("predictions")

    if not isinstance(predictions, list):
        raise RuntimeError(
            "Prediction list is missing. "
            "Check your Roboflow workflow output."
        )

    detections = []

    for p in predictions:

        confidence = float(p["confidence"])

        if confidence < min_conf:
            continue

        x = float(p["x"])
        y = float(p["y"])
        w = float(p["width"])
        h = float(p["height"])

        detections.append({

            "class": str(p["class"]),

            "confidence": confidence,

            "box": (
                x - w / 2,
                y - h / 2,
                x + w / 2,
                y + h / 2
            )

        })

    return detections


# =========================================================
# 8. DRAW BOUNDING BOXES
# =========================================================

def draw_boxes(image, detections):

    result = image.copy().convert("RGB")

    draw = ImageDraw.Draw(result)

    colors = [
        "#FF3B30",
        "#00C853",
        "#FF9100",
        "#D500F9",
        "#00B8D4"
    ]

    try:
        font = ImageFont.truetype(
            "DejaVuSans.ttf", 18
        )
    except OSError:
        font = ImageFont.load_default()

    image_width, image_height = result.size

    for i, detection in enumerate(detections):

        x1, y1, x2, y2 = detection["box"]

        # Keep coordinates within image dimensions
        x1 = max(0, min(image_width - 1, x1))
        x2 = max(0, min(image_width - 1, x2))

        y1 = max(0, min(image_height - 1, y1))
        y2 = max(0, min(image_height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        color = colors[i % len(colors)]

        # Bounding box
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=4
        )

        label = (
            f'{detection["class"]} '
            f'{detection["confidence"]:.2%}'
        )

        bbox = draw.textbbox(
            (0, 0),
            label,
            font=font
        )

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        label_y = max(0, y1 - text_height - 12)

        draw.rectangle(
            [
                x1,
                label_y,
                min(image_width, x1 + text_width + 12),
                label_y + text_height + 10
            ],
            fill=color
        )

        draw.text(
            (x1 + 5, label_y + 4),
            label,
            fill="white",
            font=font
        )

    return result


# =========================================================
# 9. SIDEBAR SETTINGS
# =========================================================

with st.sidebar:

    st.header("⚙️ Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05
    )

    st.caption(
        "Default threshold 0.30 matches your Kaggle notebook."
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Model:** RF-DETR Small")
    st.write("**Platform:** Roboflow")
    st.write("**Inference:** Cloud Workflow")

    st.divider()

    st.caption(
        "Images are processed temporarily. "
        "This application does not automatically save snapshots."
    )


# =========================================================
# 10. INPUT SELECTION
# =========================================================

st.subheader("📸 Select Image Input")

input_method = st.radio(
    "Choose your preferred input method",
    ["Camera", "Image Upload"],
    horizontal=True
)

uploaded_image = None

if input_method == "Camera":

    uploaded_image = st.camera_input(
        "Capture an image using your camera"
    )

else:

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        help="Select a JPG or PNG image."
    )


# =========================================================
# 11. DISPLAY INPUT AND DETECTION
# =========================================================

if uploaded_image is not None:

    try:

        original_image = Image.open(
            uploaded_image
        ).convert("RGB")

        st.subheader("Selected Image")

        st.image(
            original_image,
            use_container_width=True
        )

        st.caption(
            f"Image dimensions: "
            f"{original_image.width} × {original_image.height}"
        )

        if st.button(
            "🔍 Detect Weapons",
            type="primary",
            use_container_width=True
        ):

            temp_path = None

            try:

                with st.spinner(
                    "Analyzing image with RF-DETR..."
                ):

                    # Temporary image file only
                    with tempfile.NamedTemporaryFile(
                        suffix=".jpg",
                        delete=False
                    ) as temp:

                        temp_path = temp.name

                        original_image.save(
                            temp_path,
                            format="JPEG",
                            quality=95
                        )

                    detections = detect_weapons(
                        temp_path,
                        min_conf=confidence
                    )

                    result_image = draw_boxes(
                        original_image,
                        detections
                    )

                # =====================================
                # DISPLAY RESULTS
                # =====================================

                st.divider()

                st.subheader("📊 Detection Results")

                total = len(detections)

                knife_detections = [
                    d for d in detections
                    if "knife" in d["class"].lower()
                ]

                weapon_count = len(detections)
                knife_count = len(knife_detections)

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Total Detections",
                        weapon_count
                    )

                with col2:

                    st.metric(
                        "Knife Detections",
                        knife_count
                    )

                with col3:

                    st.metric(
                        "Confidence Threshold",
                        f"{confidence:.0%}"
                    )

                # =====================================
                # RESULT IMAGE
                # =====================================

                st.subheader("🎯 Detection Visualization")

                st.image(
                    result_image,
                    use_container_width=True
                )

                # =====================================
                # DETECTION STATUS
                # =====================================

                if knife_count > 0:

                    st.error(
                        f"⚠️ {knife_count} knife detection(s) found."
                    )

                elif total > 0:

                    st.warning(
                        "Other trained object classes detected. "
                        "No knife was identified."
                    )

                else:

                    st.success(
                        "No objects detected above the "
                        "selected confidence threshold."
                    )

                # =====================================
                # DETECTION DETAILS
                # =====================================

                if detections:

                    st.subheader("Detection Details")

                    for index, detection in enumerate(
                        detections, start=1
                    ):

                        with st.expander(
                            f'{index}. {detection["class"]} '
                            f'({detection["confidence"]:.2%})'
                        ):

                            st.write(
                                "**Class:**",
                                detection["class"]
                            )

                            st.write(
                                "**Confidence:**",
                                f'{detection["confidence"]:.4f}'
                            )

                            st.write(
                                "**Bounding Box (XYXY):**",
                                tuple(
                                    round(v, 2)
                                    for v in detection["box"]
                                )
                            )

                    # Download only when explicitly requested
                    from io import BytesIO

                    buffer = BytesIO()

                    result_image.save(
                        buffer,
                        format="JPEG",
                        quality=95
                    )

                    st.download_button(
                        "⬇️ Download Detection Result",
                        data=buffer.getvalue(),
                        file_name="weapon_detection_result.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )

                else:

                    st.info(
                        "Try a different image or confidence threshold "
                        "if you expected an object to be detected."
                    )

            except Exception as error:

                st.error(
                    "Detection failed. Please check your "
                    "Roboflow configuration and workflow."
                )

                st.exception(error)

            finally:

                # Remove temporary file
                if temp_path and os.path.exists(temp_path):

                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

    except Exception as error:

        st.error(
            f"Unable to read the uploaded image: {error}"
        )


# =========================================================
# 12. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#8995a6;">
        AI Weapon Detection System |
        RF-DETR Small |
        Powered by Roboflow
    </div>
    """,
    unsafe_allow_html=True
)

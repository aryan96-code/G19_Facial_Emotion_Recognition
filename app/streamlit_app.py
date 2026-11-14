"""
streamlit_app.py
================
G19 Facial Emotion Recognition — CSE276

Purpose:
    Interactive local demo that lets a user upload a face image (or use their
    webcam) and see the predicted emotion in real time.

How to run:
    streamlit run app/streamlit_app.py

Requirements:
    - A trained model file must exist in models/exp02_02E_headB_baseline_full_cw.keras
      (or fallback models/best_model.keras)
    - Run from the project root (G19_Facial_Emotion_Recognition/)

Note:
    This app is intentionally kept separate from Google Colab.
    Training happens in Colab notebooks; the trained weights are then
    downloaded to models/ and loaded here for inference only.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure src/ is on the Python path when running from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import streamlit as st

from src.preprocessing import preprocess_image
from src.utils import load_model_safe
from src.data_loader import EMOTION_LABELS
from src.face_detector import detect_and_crop_primary_face


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Facial Emotion Recognition — G19",
    page_icon="😊",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Emotion display config  (emoji + colour per class)
# ---------------------------------------------------------------------------

EMOTION_CONFIG = {
    "Angry"    : {"emoji": "😠", "color": "#FF4B4B"},
    "Disgust"  : {"emoji": "🤢", "color": "#7B2D00"},
    "Fear"     : {"emoji": "😨", "color": "#7F00FF"},
    "Happy"    : {"emoji": "😄", "color": "#FFD700"},
    "Sad"      : {"emoji": "😢", "color": "#1E90FF"},
    "Surprise" : {"emoji": "😲", "color": "#FF8C00"},
    "Neutral"  : {"emoji": "😐", "color": "#808080"},
}

PRIMARY_MODEL_PATH = (
    PROJECT_ROOT / "models" / "exp02_02E_headB_baseline_full_cw.keras"
)
FALLBACK_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"


def get_active_model_path() -> Path | None:
    """
    Resolve model path with priority:
    1. Primary: Experiment 02-E Head B model (exp02_02E_headB_baseline_full_cw.keras)
    2. Fallback: Baseline model (best_model.keras)
    """
    if PRIMARY_MODEL_PATH.exists():
        return PRIMARY_MODEL_PATH
    if FALLBACK_MODEL_PATH.exists():
        return FALLBACK_MODEL_PATH
    return None


MODEL_PATH = get_active_model_path() or PRIMARY_MODEL_PATH

# ---------------------------------------------------------------------------
# Load model (cached across Streamlit reruns)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model weights …")
def get_model():
    """Load the trained Keras model once and cache it in memory."""
    active_path = get_active_model_path()
    if active_path is None:
        raise FileNotFoundError(
            f"No valid model file found. Expected primary model at '{PRIMARY_MODEL_PATH}' "
            f"or fallback model at '{FALLBACK_MODEL_PATH}'."
        )
    return load_model_safe(str(active_path))


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict_emotion(
    image_array: np.ndarray,
    use_clahe: bool = False,
) -> tuple[str, float, np.ndarray]:
    """
    Run inference on a single face image.

    Parameters
    ----------
    image_array : np.ndarray
        Raw image (H, W, 3) or (H, W) in uint8.
    use_clahe : bool, optional
        Must match the preprocessing used during training (False for baseline).

    Returns
    -------
    label      : str    — predicted emotion label
    confidence : float  — confidence score [0, 1]
    probs      : np.ndarray — softmax probabilities for all 7 classes
    """
    model = get_model()
    processed = preprocess_image(image_array, use_clahe=use_clahe)   # → (1, 48, 48, 1)
    probs     = model.predict(processed, verbose=0)[0]
    class_idx = int(np.argmax(probs))
    label     = EMOTION_LABELS[class_idx]
    confidence = float(probs[class_idx])
    return label, confidence, probs


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

def main():
    # Header
    st.title("😊 Facial Emotion Recognition")
    st.markdown(
        "**CSE276 — Group 19** | Aryan Priyadarshi · Yuvraj · Mashwin Sharma"
    )
    st.divider()

    # --- Sidebar ---
    st.sidebar.header("ℹ️ About")
    st.sidebar.markdown(
        """
        This application uses a **Convolutional Neural Network** trained on the
        **FER-2013** dataset to classify facial expressions into one of seven
        emotion categories.

        **Selected Model:** Experiment 02-E Head B (Four-block CNN with
        Global Average Pooling + Global Max Pooling + Concatenation).

        **Emotion classes:**
        - 😠 Angry
        - 🤢 Disgust
        - 😨 Fear
        - 😄 Happy
        - 😢 Sad
        - 😲 Surprise
        - 😐 Neutral
        """
    )
    st.sidebar.divider()
    use_clahe = st.sidebar.checkbox(
        "Use CLAHE preprocessing",
        value=False,
        help="Enable only if the loaded model was trained with the optional CLAHE experiment.",
    )

    active_path = get_active_model_path()
    if active_path == PRIMARY_MODEL_PATH:
        st.sidebar.caption(
            "Model: Selected Experiment 02-E Head B (`exp02_02E_headB_baseline_full_cw.keras`)"
        )
    elif active_path == FALLBACK_MODEL_PATH:
        st.sidebar.caption(
            "Model: Fallback Baseline (`best_model.keras`)"
        )
    else:
        st.sidebar.caption("Model: Experiment 02-E Head B (not loaded)")

    # --- Model status check ---
    if active_path is None:
        st.error(
            "⚠️ **Model file not found.**\n\n"
            f"Expected primary model: `{PRIMARY_MODEL_PATH.name}`\n"
            f"Expected fallback model: `{FALLBACK_MODEL_PATH.name}`\n\n"
            f"Please place `{PRIMARY_MODEL_PATH.name}` (or `{FALLBACK_MODEL_PATH.name}`) "
            "inside the `models/` directory."
        )
        st.stop()

    # --- Input selection ---
    input_method = st.radio(
        "Choose input method:",
        ("📂 Upload an image", "📷 Use webcam"),
        horizontal=True,
    )
    st.divider()

    image_array = None

    # -- Upload mode --
    if input_method == "📂 Upload an image":
        uploaded_file = st.file_uploader(
            "Upload an image containing a face (JPG / PNG / JPEG)",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file is not None:
            # Read the uploaded file into a NumPy array
            file_bytes  = np.frombuffer(uploaded_file.read(), np.uint8)
            image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image_array is None:
                st.error("Could not decode the uploaded file. Please try a different image.")

    # -- Webcam mode --
    else:
        st.info("📷 Capture a photo using your webcam below.")
        webcam_image = st.camera_input("Take a photo")

        if webcam_image is not None:
            file_bytes  = np.frombuffer(webcam_image.read(), np.uint8)
            image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image_array is None:
                st.error("Could not decode webcam capture. Please try again.")

    # --- Face Detection & Prediction ---
    if image_array is not None:
        st.divider()
        st.subheader("Face Detection & Emotion Analysis")

        detection_result = detect_and_crop_primary_face(image_array)

        # Handle environment or classifier failure gracefully
        if detection_result.error_message:
            st.error(
                f"⚠️ **Face detection unavailable:** {detection_result.error_message}\n\n"
                "Please verify that OpenCV is installed properly (`opencv-python>=4.8.0,<5.0.0`)."
            )
            st.stop()

        # Display annotated image with bounding boxes
        col_img, col_crop = st.columns(2)
        with col_img:
            st.image(
                cv2.cvtColor(detection_result.annotated_image, cv2.COLOR_BGR2RGB),
                caption="Detected Faces (Green: Primary, Blue: Secondary)",
                use_container_width=True,
            )

        # Halt if no face was detected
        if not detection_result.detected or detection_result.cropped_face is None:
            st.warning(
                "⚠️ **No face detected in the image.**\n\n"
                "The emotion classifier requires a clearly visible frontal face. "
                "Please upload or capture a clearer, well-lit frontal face photo."
            )
            st.stop()

        # Display cropped primary face
        with col_crop:
            st.image(
                cv2.cvtColor(detection_result.cropped_face, cv2.COLOR_BGR2RGB)
                if detection_result.cropped_face.ndim == 3
                else detection_result.cropped_face,
                caption=f"Cropped Primary Face ({detection_result.cropped_face.shape[1]}×{detection_result.cropped_face.shape[0]})",
                width=160,
            )

        # Inform user if multiple faces detected
        if detection_result.num_faces > 1:
            st.info(
                f"ℹ️ **Multiple faces detected ({detection_result.num_faces}).** "
                "Analyzing the primary (largest) face highlighted with the green bounding box."
            )

        # Run inference strictly on the cropped face
        with st.spinner("Analysing emotion …"):
            try:
                label, confidence, probs = predict_emotion(
                    detection_result.cropped_face,
                    use_clahe=use_clahe,
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

        cfg   = EMOTION_CONFIG.get(label, {"emoji": "❓", "color": "#000000"})
        emoji = cfg["emoji"]
        color = cfg["color"]

        # --- Result card ---
        st.markdown(
            f"""
            <div style="
                background-color: {color}22;
                border-left: 6px solid {color};
                padding: 16px 20px;
                border-radius: 8px;
                margin-bottom: 16px;
            ">
                <h2 style="margin: 0; color: {color};">{emoji} {label}</h2>
                <p style="margin: 4px 0 0; font-size: 1rem;">
                    Confidence: <strong>{confidence * 100:.1f}%</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Probability bar chart ---
        st.markdown("#### Probability Distribution")
        class_names = [EMOTION_LABELS[i] for i in range(7)]
        prob_dict   = {name: float(probs[i]) for i, name in enumerate(class_names)}

        # Sort descending
        sorted_items = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        for emotion_name, prob in sorted_items:
            emoji_icon = EMOTION_CONFIG.get(emotion_name, {}).get("emoji", "")
            st.progress(prob, text=f"{emoji_icon} {emotion_name}: {prob * 100:.1f}%")

        st.divider()
        st.caption(
            "⚠️ This is an academic demo. Results depend on face visibility, "
            "image quality, and lighting conditions."
        )


if __name__ == "__main__":
    main()

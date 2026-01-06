import io
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from model_factory import build_model
from transforms import build_eval_transform
from config import (
    CLASSES,
    IMG_SIZE,
    DEFAULT_WEIGHTS_PATH,
)
# --- Config / constants (keep in sync with train/test) ---
ROOT = Path(__file__).resolve().parents[1]

# --- Transforms (same idea as train/test, but for single images) ---
eval_transform = build_eval_transform(img_size=IMG_SIZE)

# --- Model loading ---
@st.cache_resource
def load_model(weights_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=len(CLASSES))
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, device

def predict_image(model, device, img: Image.Image):
    # img: PIL Image
    tensor = eval_transform(img).unsqueeze(0).to(device)  # shape [1, C, H, W]

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().squeeze(0)

    conf, idx = torch.max(probs, dim=0)
    pred_class = CLASSES[int(idx)]
    return pred_class, float(conf), probs.tolist()

# --- Streamlit UI ---
def main():
    st.set_page_config(
        page_title="Face Mask Detection Demo",
        page_icon="🧾",
        layout="centered",
    )

    st.title("Face Mask Detection Demo")
    st.write(
        "Upload an image or capture one from your camera. "
        "The model will predict whether the person is **with_mask** or **without_mask**."
    )

    # Check weights exist
    if not DEFAULT_WEIGHTS_PATH.exists():
        st.error(
            f"Model weights not found at `{DEFAULT_WEIGHTS_PATH}`.\n\n"
            "Place your `best_mobilenetv2.pt` file in the `models/` folder."
        )
        return

    model, device = load_model(DEFAULT_WEIGHTS_PATH)
    st.caption(f"Running on device: `{device}`")

    st.subheader("1. Provide an image")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader(
            "Upload a face image",
            type=["jpg", "jpeg", "png"],
            help="Upload a close-up face image with or without a mask.",
        )

    with col2:
        camera_image = st.camera_input(
            "Or capture from your camera",
            help="Use your device camera to take a photo.",
        )

    image_source = None
    source_label = None

    if camera_image is not None:
        # Prefer camera image if provided
        image_source = camera_image
        source_label = "camera"
    elif uploaded_file is not None:
        image_source = uploaded_file
        source_label = "upload"

    if image_source is None:
        st.info("👆 Upload an image or capture from your camera to get a prediction.")
        return

    # Load the image from the selected source
    if source_label == "camera":
        bytes_data = image_source.getvalue()
    else:
        bytes_data = image_source.read()

    img = Image.open(io.BytesIO(bytes_data))

    st.subheader("2. Preview")
    st.image(img, caption=f"Input image ({source_label})", width='content')

    if st.button("Run prediction"):
        with st.spinner("Running model..."):
            pred_class, confidence, all_probs = predict_image(model, device, img)

        st.subheader("3. Result")
        st.markdown(
            f"**Prediction:** `{pred_class}`  \n"
            f"**Confidence:** {confidence:.2%}"
        )

        st.json({
            "classes": CLASSES,
            "probabilities": {CLASSES[i]: round(p, 4) for i, p in enumerate(all_probs)},
        })

if __name__ == "__main__":
    main()

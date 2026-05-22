import os
import sys
import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

from src.models.baseline_cnn import BaselineCNN
from src.models.resnet50 import build_resnet50
from src.models.efficientnet_b3 import build_efficientnet_b3
from src.explainability.gradcam import GradCAM
from src.training.lora import apply_lora

# =========================
# CONFIG
# =========================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

st.set_page_config(
    page_title="Industrial Defect Detection AI",
    page_icon="🛰️",
    layout="wide"
)

# =========================
# HEADER
# =========================

st.title("🛰️ Industrial Surface Defect Detection System")
st.caption("AI-powered visual inspection system for aerospace-grade material quality control")

st.markdown("""
### 🧠 System Overview
Automated defect detection system for industrial surfaces using deep learning models trained on NEU-DET dataset.
""")

# =========================
# SIDEBAR
# =========================

st.sidebar.header("⚙️ Settings")

model_option = st.sidebar.selectbox(
    "Select Model",
    ["ResNet50", "EfficientNet-B3", "Baseline CNN"],
    index=0
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.sidebar.markdown("### System Status")
st.sidebar.success("Model Ready")
st.sidebar.info(f"Device: {DEVICE}")

show_gradcam = st.sidebar.checkbox("Show Grad-CAM", value=True)

# =========================
# CLASSES
# =========================

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

CLASS_DESCRIPTIONS = {
    "crazing": "Network of fine cracks on the surface",
    "inclusion": "Foreign material embedded in the surface",
    "patches": "Irregular patches on the material",
    "pitted_surface": "Small pits or cavities",
    "rolled-in_scale": "Scale pressed into surface",
    "scratches": "Linear grooves or marks"
}

# =========================
# MODEL LOADING (NO LORA)
# =========================

@st.cache_resource
def load_model(model_name):

    if model_name == "ResNet50":
        model = build_resnet50(num_classes=6, freeze_backbone=False)
        model = apply_lora(
            model,
            r=4,
            alpha=32,
            dropout=0.1,
            target_modules=["fc"]
        )
        checkpoint = "checkpoints/best_resnet50.pt"

    elif model_name == "EfficientNet-B3":
        model = build_efficientnet_b3(num_classes=6, freeze_backbone=False)
        model = apply_lora(
            model,
            r=4,
            alpha=32,
            dropout=0.1,
            target_modules=["classifier"]
        )
        checkpoint = "checkpoints/best_efficientnet_b3.pt"

    else:
        model = BaselineCNN(num_classes=6)
        checkpoint = "checkpoints/best_baseline_cnn.pt"

    state_dict = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE)
    model.eval()

    return model

# =========================
# TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# PREDICTION
# =========================

def predict(image, model):
    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return pred.item(), conf.item(), probs.cpu().numpy()[0]

# =========================
# GRAD-CAM
# =========================

def generate_gradcam(image, model, class_idx):
    try:
        gradcam = GradCAM(
            model,
            target_layer=model.layer4[-1] if hasattr(model, "layer4") else model.features[-1]
        )

        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            model(img_tensor)

        heatmap = gradcam.generate_cam(img_tensor, class_idx)
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        original = np.array(image.resize((224, 224)))
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        return overlay

    except Exception:
        return None

# =========================
# UI
# =========================

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📤 Upload Image")

    file = st.file_uploader("Upload steel surface image", type=["jpg", "jpeg", "png"])

    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, use_container_width=True)

        if st.button("Run Inspection 🛰️", type="primary"):
            with st.spinner("Processing..."):

                model = load_model(model_option)

                pred, conf, probs = predict(image, model)
                class_name = CLASS_NAMES[pred]

                st.success("Done")

                with col2:
                    st.subheader("📊 Results")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Class", class_name)
                    c2.metric("Confidence", f"{conf:.2%}")
                    c3.metric("Model", model_option)

                    st.info(CLASS_DESCRIPTIONS[class_name])

                    # probabilities
                    st.subheader("Probability Distribution")

                    fig, ax = plt.subplots()
                    ax.bar(CLASS_NAMES, probs)
                    ax.set_ylim(0, 1)
                    plt.xticks(rotation=45)

                    st.pyplot(fig)

                    # Grad-CAM
                    if show_gradcam:
                        st.subheader("Explainability (Grad-CAM)")

                        cam = generate_gradcam(image, model, pred)

                        if cam is not None:
                            st.image(cam, use_container_width=True)
                        else:
                            st.warning("Grad-CAM failed")

# =========================
# FOOTER
# =========================

st.markdown("---")
st.markdown("AI Industrial Inspection System • Streamlit Demo")
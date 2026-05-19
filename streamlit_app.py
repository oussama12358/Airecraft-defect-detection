import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import sys
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.models.baseline_cnn import BaselineCNN
from src.models.resnet50 import build_resnet50
from src.models.efficientnet_b3 import build_efficientnet_b3
from src.explainability.gradcam import GradCAM
from omegaconf import OmegaConf

# Page config
st.set_page_config(
    page_title="Steel Surface Defect Detection",
    page_icon="🔧",
    layout="wide"
)

# Title
st.title("🔧 Steel Surface Defect Detection")
st.markdown("Upload an image to detect and classify surface defects in steel materials.")

# Sidebar
st.sidebar.header("Settings")

# Model selection
model_option = st.sidebar.selectbox(
    "Select Model",
    ["ResNet50", "EfficientNet-B3", "Baseline CNN"],
    index=0
)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Device: {DEVICE}")

# Class names
CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_DESCRIPTIONS = {
    "crazing": "Network of fine cracks on the surface",
    "inclusion": "Foreign material embedded in the surface",
    "patches": "Irregular patches on the material",
    "pitted_surface": "Small pits or cavities on the surface",
    "rolled-in_scale": "Scale pressed into the surface during rolling",
    "scratches": "Linear marks or grooves on the surface"
}

# Load model
@st.cache_resource
def load_model(model_name):
    """Load the selected model."""
    cfg = OmegaConf.to_container(OmegaConf.load("configs/config.yaml"), resolve=True)
    
    if model_name == "ResNet50":
        model = build_resnet50(num_classes=6, freeze_backbone=False)
        checkpoint = "checkpoints/best_resnet50.pt"
    elif model_name == "EfficientNet-B3":
        model = build_efficientnet_b3(num_classes=6, freeze_backbone=False)
        checkpoint = "checkpoints/best_efficientnet_b3.pt"
    else:  # Baseline CNN
        model = BaselineCNN(num_classes=6)
        checkpoint = "checkpoints/best_baseline_cnn.pt"
    
    # Apply LoRA if configured
    if cfg["training"].get("use_lora", False):
        from src.training.lora import apply_lora
        model = apply_lora(
            model, 
            r=cfg["training"]["lora_rank"], 
            alpha=cfg["training"]["lora_alpha"], 
            dropout=cfg["training"]["lora_dropout"],
            target_modules=cfg["training"].get("lora_target_modules", ["fc", "classifier"])
        )
    
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Predict function
def predict(image, model):
    """Make prediction on the image."""
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]

# Grad-CAM function
def generate_gradcam(image, model, predicted_class):
    """Generate Grad-CAM heatmap."""
    try:
        gradcam = GradCAM(model, target_layer=model.layer4[-1] if hasattr(model, 'layer4') else model.features[-1])
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(image_tensor)
        
        heatmap = gradcam.generate_cam(image_tensor, predicted_class)
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        original = np.array(image.resize((224, 224)))
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
        
        return overlay
    except Exception as e:
        st.warning(f"Grad-CAM generation failed: {e}")
        return None

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Predict button
        if st.button("Detect Defect", type="primary"):
            with st.spinner("Analyzing image..."):
                # Load model
                model = load_model(model_option)
                
                # Make prediction
                predicted_class, confidence, probabilities = predict(image, model)
                
                # Display results
                st.success("Analysis Complete!")
                
                with col2:
                    st.subheader("Prediction Results")
                    
                    # Main prediction
                    class_name = CLASS_NAMES[predicted_class]
                    st.metric(
                        "Predicted Class",
                        class_name,
                        f"{confidence:.2%} confidence"
                    )
                    
                    # Description
                    st.info(f"**Description:** {CLASS_DESCRIPTIONS[class_name]}")
                    
                    # Probability chart
                    st.subheader("Class Probabilities")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    bars = ax.bar(CLASS_NAMES, probabilities, color='steelblue')
                    bars[predicted_class].set_color('coral')
                    ax.set_xlabel('Defect Class')
                    ax.set_ylabel('Probability')
                    ax.set_title('Prediction Probabilities')
                    ax.set_ylim(0, 1)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Grad-CAM
                    st.subheader("Grad-CAM Heatmap")
                    gradcam_image = generate_gradcam(image, model, predicted_class)
                    if gradcam_image is not None:
                        st.image(gradcam_image, caption="Grad-CAM Heatmap (Red = High Attention)", use_column_width=True)
                    else:
                        st.info("Grad-CAM not available for this model")

# Instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Upload an image** of a steel surface
    2. **Select a model** from the sidebar (ResNet50 recommended)
    3. **Click "Detect Defect"** to analyze the image
    4. **View results** including:
       - Predicted defect class
       - Confidence score
       - Probability distribution
       - Grad-CAM heatmap (shows which regions the model focused on)
    """)

# Model info
with st.expander("📊 Model Information"):
    st.markdown("""
    **Available Models:**
    
    - **ResNet50**: Best overall performance (~99-100% accuracy)
    - **EfficientNet-B3**: More robust to blur and noise
    - **Baseline CNN**: Simpler architecture, faster inference
    
    **Defect Classes:**
    - `crazing`: Fine cracks
    - `inclusion`: Foreign materials
    - `patches`: Irregular patches
    - `pitted_surface`: Small pits
    - `rolled-in_scale`: Rolling scale
    - `scratches`: Linear marks
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Steel Surface Defect Detection")

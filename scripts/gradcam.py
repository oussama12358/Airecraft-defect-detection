import os
import sys
import torch
import cv2
import numpy as np
import argparse
from torchvision import models, transforms
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.models.baseline_cnn import BaselineCNN
from src.models.resnet50 import build_resnet50
from src.models.efficientnet_b3 import build_efficientnet_b3

# =========================
# ARGUMENTS
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--image", type=str, required=True)
parser.add_argument("--output", type=str, default="assets/gradcam.jpg")
args = parser.parse_args()

MODEL_PATH = args.model
IMG_PATH = args.image
OUTPUT_PATH = args.output

# =========================
# BUILD MODEL FROM CHECKPOINT NAME
# =========================
model_name = os.path.basename(MODEL_PATH).lower()
if "baseline_cnn" in model_name:
    model = BaselineCNN(num_classes=6)
elif "resnet50" in model_name:
    model = build_resnet50(num_classes=6, freeze_backbone=False)
elif "efficientnet_b3" in model_name:
    model = build_efficientnet_b3(num_classes=6, freeze_backbone=False)
else:
    raise ValueError(
        "Unknown checkpoint name. Use a checkpoint filename containing "
        "baseline_cnn, resnet50, or efficientnet_b3."
    )

state_dict = torch.load(MODEL_PATH, map_location="cpu")
if isinstance(state_dict, dict) and "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]

model.load_state_dict(state_dict)
model.eval()

# =========================
# HOOKS
# =========================
gradients = None
activations = None

def save_gradients(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

def save_activations(module, input, output):
    global activations
    activations = output

target_layer = model.layer4[-1]
target_layer.register_forward_hook(save_activations)
target_layer.register_full_backward_hook(save_gradients)

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# LOAD IMAGE
# =========================
image = Image.open(IMG_PATH).convert("RGB")
x = transform(image).unsqueeze(0)

# =========================
# FORWARD
# =========================
output = model(x)
pred_class = output.argmax(dim=1)

# =========================
# BACKWARD
# =========================
model.zero_grad()
output[0, pred_class].backward()

# =========================
# GRAD-CAM
# =========================
pooled = torch.mean(gradients, dim=[0, 2, 3])

for i in range(activations.shape[1]):
    activations[:, i, :, :] *= pooled[i]

heatmap = torch.mean(activations, dim=1).squeeze()
heatmap = torch.relu(heatmap)
heatmap = heatmap.detach().numpy()
heatmap /= heatmap.max()

# =========================
# OVERLAY
# =========================
img = cv2.imread(IMG_PATH)

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

result = heatmap * 0.4 + img

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
cv2.imwrite(OUTPUT_PATH, result)

print("✅ Grad-CAM saved:", OUTPUT_PATH)
import os
import sys
import torch
import cv2
import numpy as np
import argparse
from omegaconf import OmegaConf
from torchvision import transforms
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.models.baseline_cnn import BaselineCNN
from src.models.resnet50 import build_resnet50
from src.models.efficientnet_b3 import build_efficientnet_b3
from src.training.lora import apply_lora

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--image", required=True)
parser.add_argument("--output", default="assets/gradcam.jpg")
args = parser.parse_args()

MODEL_PATH = args.model
IMG_PATH = args.image
OUTPUT_PATH = args.output

# Build the correct model architecture for the checkpoint
checkpoint_name = os.path.basename(MODEL_PATH).lower()
if "baseline_cnn" in checkpoint_name:
    model = BaselineCNN(num_classes=6)
elif "resnet50" in checkpoint_name:
    model = build_resnet50(num_classes=6, freeze_backbone=False)
elif "efficientnet_b3" in checkpoint_name:
    model = build_efficientnet_b3(num_classes=6, freeze_backbone=False)
else:
    raise ValueError(
        "Unknown checkpoint name. Use a checkpoint filename containing "
        "baseline_cnn, resnet50, or efficientnet_b3."
    )

cfg = OmegaConf.to_container(OmegaConf.load("configs/config.yaml"), resolve=True)
if cfg["training"].get("use_lora", False):
    model = apply_lora(
        model,
        r=cfg["training"]["lora_rank"],
        alpha=cfg["training"]["lora_alpha"],
        dropout=cfg["training"]["lora_dropout"],
        target_modules=cfg["training"].get("lora_target_modules", ["fc", "classifier"]),
    )

state = torch.load(MODEL_PATH, map_location="cpu")
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]

model.load_state_dict(state)
model.eval()

gradients = None
activations = None

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

def forward_hook(module, inp, out):
    global activations
    activations = out

# Use the last ResNet block or the last EfficientNet block
if "resnet50" in checkpoint_name:
    target_layer = model.layer4[-1]
elif "efficientnet_b3" in checkpoint_name:
    target_layer = list(model.features.children())[-1]
else:
    target_layer = model.features[-1]

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open(IMG_PATH).convert("RGB")
x = transform(image).unsqueeze(0)

output = model(x)
pred_class = output.argmax(dim=1)

model.zero_grad()
output[0, pred_class].backward()

weights = torch.mean(gradients, dim=[0, 2, 3])
for i in range(activations.shape[1]):
    activations[:, i, :, :] *= weights[i]

heatmap = torch.mean(activations, dim=1).squeeze().detach().numpy()
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

img = cv2.imread(IMG_PATH)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

result = heatmap * 0.4 + img
result = np.clip(result, 0, 255).astype(np.uint8)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
cv2.imwrite(OUTPUT_PATH, result)
print(f"Saved GradCAM: {OUTPUT_PATH}")
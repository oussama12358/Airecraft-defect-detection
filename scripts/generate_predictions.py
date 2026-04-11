import csv
import os
import sys
import torch
import cv2
import argparse
from PIL import Image
from torchvision import transforms

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
parser.add_argument("--test_dir", type=str, default="data/test")
parser.add_argument("--split_csv", type=str, default="data/splits/test.csv")
parser.add_argument("--img_dir", type=str, default="data/processed/images")
parser.add_argument("--output_dir", type=str, default="assets/results")
args = parser.parse_args()

MODEL_PATH = args.model
TEST_DIR = args.test_dir
SPLIT_CSV = args.split_csv
IMG_DIR = args.img_dir
OUTPUT_DIR = args.output_dir

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# CLASS NAMES (NEU-DET)
# =========================
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# BUILD MODEL FROM CHECKPOINT NAME
# =========================
def build_model(checkpoint_path):
    model_name = os.path.basename(checkpoint_path).lower()

    if "baseline_cnn" in model_name:
        model = BaselineCNN(num_classes=6)
    elif "resnet50" in model_name:
        model = build_resnet50(num_classes=6, freeze_backbone=False)
    elif "efficientnet_b3" in model_name:
        model = build_efficientnet_b3(num_classes=6, freeze_backbone=False)
    else:
        raise ValueError(
            "Unknown checkpoint name. Use checkpoint filename containing "
            "baseline_cnn, resnet50, or efficientnet_b3."
        )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model.eval()
    return model

# =========================
# LOAD MODEL
# =========================
model = build_model(MODEL_PATH)

# =========================
# PREDICT FUNCTION
# =========================
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return pred.item(), conf.item()

# =========================
# BUILD IMAGE LIST
# =========================
image_paths = []
if os.path.isdir(TEST_DIR):
    for img_name in sorted(os.listdir(TEST_DIR)):
        if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            image_paths.append(os.path.join(TEST_DIR, img_name))
elif os.path.isfile(SPLIT_CSV):
    with open(SPLIT_CSV, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = row.get("filename") or row.get("file") or row.get("image")
            if not filename:
                continue
            image_paths.append(os.path.join(IMG_DIR, filename))
else:
    raise FileNotFoundError(
        f"No test directory found at {TEST_DIR} and no split CSV found at {SPLIT_CSV}."
    )

if not image_paths:
    raise RuntimeError("No test images found for prediction.")

# =========================
# RUN ON TEST SET
# =========================
for path in image_paths:
    if not os.path.isfile(path):
        print(f"Skipping missing file: {path}")
        continue

    pred, conf = predict(path)

    img = cv2.imread(path)
    img = cv2.resize(img, (500, 500))

    label = f"{CLASS_NAMES[pred]} ({conf:.2f})"
    filename = os.path.basename(path)

    cv2.putText(
        img,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"pred_{filename}"), img)

print("✅ Predictions completed successfully!")
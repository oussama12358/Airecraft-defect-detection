# ✈️ Aircraft Defect Detection using Deep Learning

> A deep learning system for automatic detection of surface defects in aircraft and industrial materials using image classification models trained on the **NEU-DET** dataset.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset & Defect Classes](#-dataset--defect-classes)
- [Models Used](#-models-used)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Output](#-output)
- [Key Features](#-key-features)
- [Use Cases](#-use-cases)
- [Author](#-author)
- [License](#-license)

---

## 📌 Project Overview

This project focuses on **detecting and classifying surface defects** in steel and industrial materials using multiple deep learning architectures. The system supports end-to-end inference: from raw image input to annotated predictions with confidence scores.

The pipeline is designed for real-world deployment in aerospace and industrial quality control contexts.

---

## 🗂️ Dataset & Defect Classes

**Dataset:** [NEU-DET](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) — a standard benchmark for surface defect detection.

The model classifies **6 types of surface defects**:

| # | Class | Description |
|---|-------|-------------|
| 1 | `crazing` | Network of fine cracks on the surface |
| 2 | `inclusion` | Foreign material embedded in the surface |
| 3 | `patches` | Irregular patches on the material |
| 4 | `pitted_surface` | Small pits or cavities on the surface |
| 5 | `rolled-in_scale` | Scale pressed into the surface during rolling |
| 6 | `scratches` | Linear marks or grooves on the surface |

---

## 🧠 Models Used

| Model | Architecture | Accuracy |
|-------|-------------|----------|
| 🥉 Baseline CNN | Custom lightweight CNN | ~95.8% |
| 🥈 EfficientNet-B3 | Transfer learning (EfficientNet) | ~99.5% |
| 🥇 **ResNet50** | **Transfer learning (ResNet50)** | **100%** |

> ✅ **Best model: ResNet50** — achieves perfect classification on the NEU-DET test set.

---

## 📊 Results

- **Best model:** ResNet50 — 100% accuracy
- **Evaluation metrics:** Confusion matrix + per-class classification accuracy
- **Generalization:** Strong performance across all 6 defect categories on the NEU-DET benchmark

---

## 📁 Project Structure

```
aircraft-defect-detection/
│
├── checkpoints/                  # Saved model weights
│   ├── best_resnet50.pt
│   ├── best_efficientnet_b3.pt
│   └── best_baseline_cnn.pt
│
├── scripts/
│   ├── download_data.py     # Download NEU-DET dataset
│   ├── prepare_splits.py    # Create train/val/test CSV splits
│   ├── export_onnx.py       # Export checkpoint to ONNX
│   ├── generate_predictions.py  # Batch prediction images
│   └── gradcam.py           # Generate Grad-CAM heatmap for one image
├── src/
│   ├── datasets/            # Dataset and transform utilities
│   ├── evaluation/          # Metrics and reporting code
│   ├── explainability/      # GradCAM helper code
│   ├── models/              # Model definitions
│   └── training/            # Trainer and scheduler
├── static/                  # Static web assets
├── train.py                 # Model training entrypoint
├── evaluate.py              # Evaluation entrypoint
├── requirements.txt         # Python dependencies
└── mlflow.db                # MLflow tracking database
```

## Quickstart (Windows Native)

### 1. Prerequisites
- Install Python 3.11 from https://python.org/downloads
- Install Git from https://git-scm.com/downloads

### 2. Clone the repo
```powershell
git clone <your-repo-url>
cd "defect-detection"
```

### 3. Create a virtual environment
```powershell
python -m venv .venv_new
.venv_new\Scripts\Activate.ps1
```

### 4. Install dependencies
```powershell
pip install -r requirements.txt
```

### 5. Download the NEU-DET dataset
```powershell
python scripts/download_data.py
```
If download does not work, manually download from:
https://www.kaggle.com/datasets/uciml/neu-surface-defect-database
and place the dataset under `data/raw/NEU-DET/`.

### 6. Prepare data splits
```powershell
python scripts/prepare_splits.py
```

### 7. Train a model
Edit `configs/config.yaml` and choose one model:
```yaml
model:
  name: resnet50
```
Then run:
```powershell
python train.py
```
The best weights are saved to `checkpoints/best_{model_name}.pt`.

### 8. Evaluate a model
```powershell
python evaluate.py --checkpoint checkpoints/best_resnet50.pt
```
The evaluation script saves a report and confusion matrix to `reports/`.

### 9. Generate batch predictions
Use this script to label a folder of test images and save annotated outputs:
```powershell
python scripts/generate_predictions.py --model checkpoints/best_resnet50.pt --split_csv data/splits/test.csv --img_dir data/processed/images
```
If you have a local image folder, pass `--test_dir` instead.

### 10. Generate Grad-CAM explanation
```powershell
python scripts/gradcam.py --model checkpoints/best_resnet50.pt --image data/processed/images/inclusion_inclusion_220.jpg
```
The heatmap output is saved to `assets/gradcam.jpg` by default.

### 11. Run the API
```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Open: http://localhost:8000/docs

## API Reference

### POST /predict
Predict defect class for a single image.

**Request**: multipart form upload with field `file`.

**Response**:
```json
{
  "predicted_class": "crazing",
  "confidence": 0.9876,
  "all_probabilities": {
    "crazing": 0.9876,
    "inclusion": 0.0054,
    "patches": 0.0031,
    "pitted_surface": 0.0020,
    "rolled-in_scale": 0.0012,
    "scratches": 0.0007
  },
  "gradcam_heatmap_base64": "...",
  "latency_ms": 123.45,
  "tta_used": false
}
```

### GET /health
Returns API health and model information.

## Notes
- `reports/` contains generated evaluation files and is excluded from version control.
- `checkpoints/` contains best saved model weights.
- `mlflow.db` stores MLflow tracking data.

## License
MIT
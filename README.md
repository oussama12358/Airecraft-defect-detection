# Defect Detection System

A real ML project for industrial surface defect detection using the NEU-DET dataset. The system supports training, evaluation, inference, and explainability for three model types.

## Key Highlights
- End-to-end defect classification pipeline
- FastAPI inference service
- MLflow tracking with SQLite
- Multiple model architectures: `baseline_cnn`, `resnet50`, `efficientnet_b3`
- Evaluation reports and confusion matrices
- GradCAM explainability for visual model attention
- ONNX export support for deployment

## Project Structure
```
defect-detection/
├── api/
│   ├── main.py              # FastAPI application
│   ├── inference.py         # Inference logic
│   ├── schemas.py           # API response models
│   └── __init__.py
├── checkpoints/             # Saved PyTorch checkpoint files
│   ├── best_baseline_cnn.pt
│   ├── best_resnet50.pt
│   ├── best_efficientnet_b3.pt
│   ├── best_model.pt
│   └── model.onnx
├── configs/
│   └── config.yaml          # Training configuration
├── data/
│   ├── processed/
│   │   └── images/          # Images used for training and evaluation
│   ├── raw/
│   │   └── NEU-DET/         # Downloaded dataset
│   └── splits/              # train/val/test split CSV files
├── reports/                 # Saved evaluation reports and confusion matrices
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
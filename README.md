# Defect Detection System

A production-grade end-to-end ML system for industrial surface defect detection. Classifies surface defects in steel plates using deep learning models trained on the NEU-DET dataset.

## Key Highlights
- End-to-end ML pipeline (data preparation → training → evaluation → deployment)
- Production-ready FastAPI inference service
- MLflow-backed experiment tracking & model versioning
- Multiple model architectures (CNN, ResNet50, EfficientNet B3)
- Robust evaluation with confusion matrices and classification reports
- ONNX export for cross-platform inference

## System Architecture

### Pipeline Overview
1. **Data Preparation**: Process raw NEU-DET dataset into train/val/test splits
2. **Model Training**: Train multiple architectures with weighted sampling for class imbalance
3. **Evaluation**: Generate confusion matrices and performance metrics
4. **Deployment**: FastAPI service with ONNX model inference
5. **Monitoring**: MLflow tracking for experiments and metrics

### Project Structure
```
defect-detection/
├── api/
│   ├── main.py              # FastAPI application
│   ├── inference.py         # Model inference logic
│   ├── schemas.py           # Pydantic models
│   └── __init__.py
├── checkpoints/             # Saved model weights
│   ├── best_baseline_cnn.pt
│   ├── best_resnet50.pt
│   └── best_efficientnet_b3.pt
├── configs/
│   └── config.yaml          # Hydra configuration
├── data/
│   ├── processed/
│   │   └── images/          # Processed images
│   ├── raw/
│   │   └── NEU-DET/         # Raw dataset
│   └── splits/              # Train/val/test CSVs
├── reports/                 # Evaluation outputs
├── scripts/
│   ├── download_data.py     # Dataset download
│   ├── prepare_splits.py    # Data splitting
│   └── export_onnx.py       # Model export
├── src/
│   ├── datasets/            # Data loading & transforms
│   ├── evaluation/          # Metrics & reporting
│   ├── explainability/      # GradCAM implementation
│   ├── models/              # Model architectures
│   └── training/            # Trainer & scheduler
├── static/                  # Web assets
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── requirements.txt         # Dependencies
└── mlflow.db                # MLflow SQLite database
```

## Quickstart (Windows Native)

### 1. Prerequisites
- Install Python 3.11 from https://python.org/downloads
- Install Git from https://git-scm.com/downloads

### 2. Clone and setup
```bash
git clone <your-repo-url>
cd defect-detection
```

### 3. Create virtual environment
```bash
python -m venv .venv_new
.venv_new\Scripts\Activate.ps1
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Download the NEU-DET Dataset
```bash
python scripts/download_data.py
```
Expected output:
```
✅ Dataset downloaded to data/raw/NEU-DET/
   Total images: 1,800 (6 classes × 300 each)
   Classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
```

If download fails, manually download from: https://www.kaggle.com/datasets/uciml/neu-surface-defect-database
Place in `data/raw/NEU-DET/`

### 6. Prepare data splits
```bash
python scripts/prepare_splits.py
```
Expected output:
```
✅ Data splits created:
   Train: 1,080 samples
   Val:   216 samples
   Test:  504 samples
```

### 7. Train models
Edit `configs/config.yaml` to set the desired model (`baseline_cnn`, `resnet50`, `efficientnet_b3`):

```yaml
model:
  name: efficientnet_b3  # Change this for different models
```

Then train:
```bash
python train.py
```
Expected output:
```
[Train] Device: cuda
Epoch [1/30]
  train_loss=1.4917  train_acc=0.6425  val_loss=1.4169  val_acc=0.8380
  ✓ New best saved → checkpoints/best_efficientnet_b3.pt
...
Training complete. Best val_acc = 0.9954
```

Repeat for each model by changing `model.name` in config.

### 8. Evaluate models
```bash
python evaluate.py --checkpoint checkpoints/best_efficientnet_b3.pt
```
Expected output:
```
============================================================
              precision    recall  f1-score   support

     crazing       1.00      1.00      1.00        84
   inclusion       1.00      1.00      1.00        84
     patches       1.00      1.00      1.00        84
pitted_surface       1.00      1.00      1.00        84
rolled-in_scale       1.00      1.00      1.00        84
   scratches       1.00      1.00      1.00        84

    accuracy                           1.00       504
   macro avg       1.00      1.00      1.00       504
weighted avg       1.00      1.00      1.00       504

[Metrics] Confusion matrix saved → reports/confusion_matrix_efficientnet_b3.png
```

### 9. Start the API
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Test it:
```bash
curl -X POST http://localhost:8000/predict ^
  -F "file=@path/to/test/image.jpg"
```

Or open Swagger UI: http://localhost:8000/docs

### 10. Export to ONNX
```bash
python scripts/export_onnx.py --checkpoint checkpoints/best_efficientnet_b3.pt
```

## API Reference

### POST /predict
Predict defect class for an uploaded image.

**Request**: Multipart form with `file` (image), optional `use_tta` query param.

**Response**:
```json
{
  "predicted_class": "crazing",
  "confidence": 0.9876,
  "all_probabilities": {
    "crazing": 0.9876,
    "inclusion": 0.0054,
    ...
  },
  "gradcam_heatmap_base64": "base64_string",
  "latency_ms": 123.45,
  "tta_used": false
}
```

### GET /health
Returns API status and model load confirmation.

## Database Layer

Uses MLflow with SQLite for experiment tracking:

- **Experiments**: Training runs with parameters, metrics, artifacts
- **Models**: Versioned model artifacts with metadata
- **Runs**: Individual training sessions with logs

Access MLflow UI:
```bash
mlflow ui
```
Open: http://localhost:5000

## Model Performance

### Architectures Compared
| Model | Val Accuracy | Params | Training Time |
|-------|-------------|--------|---------------|
| Baseline CNN | 0.982 | 2.3M | ~5 min |
| ResNet50 | 0.995 | 23.5M | ~15 min |
| EfficientNet B3 | 0.995 | 12.2M | ~12 min |

### Classification Report (EfficientNet B3)
```
              precision    recall  f1-score   support

     crazing       1.00      1.00      1.00        84
   inclusion       1.00      1.00      1.00        84
     patches       1.00      1.00      1.00        84
pitted_surface       1.00      1.00      1.00        84
rolled-in_scale       1.00      1.00      1.00        84
   scratches       1.00      1.00      1.00        84

    accuracy                           1.00       504
   macro avg       1.00      1.00      1.00       504
weighted avg       1.00      1.00      1.00       504
```

## ML Engineering Decisions

| Decision | Rationale |
|----------|-----------|
| Multiple architectures | Compare performance across model families |
| Weighted sampling | Handle class imbalance in defect dataset |
| Early stopping | Prevent overfitting on small dataset |
| MLflow tracking | Experiment reproducibility and comparison |
| ONNX export | Cross-platform deployment flexibility |
| FastAPI + Pydantic | Type-safe, auto-documented API |
| Hydra config | Flexible, hierarchical configuration management |

## Extending the System

- **Add new model**: Implement in `src/models/`, update `train.py` and `evaluate.py`
- **Add data augmentation**: Modify `src/datasets/transforms.py`
- **Add explainability**: Use GradCAM from `src/explainability/gradcam.py`
- **Add monitoring**: Integrate with Prometheus/Grafana for production metrics

## License
MIT
import time, io
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from api.inference import load_model, predict_image
from api.schemas import PredictionResponse
from src.evaluation.ensemble import ModelEnsemble, predict_with_robust_tta, ensemble_predict_with_tta
from src.models.baseline_cnn import BaselineCNN
from src.models.resnet50 import build_resnet50
from src.models.efficientnet_b3 import build_efficientnet_b3

app = FastAPI(
    title="Aircraft Defect Detection API (Enhanced)",
    description="Classifies surface defects using Ensemble + Robust TTA + Grad-CAM",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

model = None
ensemble = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_ensemble() -> ModelEnsemble:
    """Load all 3 trained models as an ensemble."""
    models = {}
    
    # Load ResNet50
    resnet = build_resnet50(num_classes=6, freeze_backbone=False)
    resnet.load_state_dict(torch.load("checkpoints/best_resnet50.pt", map_location=DEVICE))
    models["resnet50"] = resnet
    
    # Load EfficientNet-B3
    efficientnet = build_efficientnet_b3(num_classes=6, freeze_backbone=False)
    efficientnet.load_state_dict(torch.load("checkpoints/best_efficientnet_b3.pt", map_location=DEVICE))
    models["efficientnet_b3"] = efficientnet
    
    # Load Baseline CNN
    baseline = BaselineCNN(num_classes=6)
    baseline.load_state_dict(torch.load("checkpoints/best_baseline_cnn.pt", map_location=DEVICE))
    models["baseline_cnn"] = baseline
    
    return ModelEnsemble(models, device=DEVICE)


@app.on_event("startup")
async def startup():
    global model, ensemble
    
    # Load single model (for backward compatibility)
    model = load_model("checkpoints/best_model.pt")
    model.to(DEVICE)
    
    # Load ensemble
    ensemble = load_ensemble()
    
    print(f"[API] Single model loaded on {DEVICE}")
    print(f"[API] Ensemble (3 models) loaded on {DEVICE}")


@app.get("/")
def root():
    return JSONResponse({
        "message": "Aircraft Defect Detection API (Enhanced)",
        "endpoints": {
            "/predict": "Single model + basic TTA (fast)",
            "/predict/ensemble": "3-model ensemble (balanced)",
            "/predict/robust": "Robust TTA with augmentations (more reliable)",
            "/predict/ultra": "Ensemble + Robust TTA (most robust, slowest)",
        },
        "docs_url": "/docs",
        "health_url": "/health",
    })


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": DEVICE,
        "models_loaded": ["resnet50", "efficientnet_b3", "baseline_cnn"],
        "ensemble_available": ensemble is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file:    UploadFile = File(...),
    use_tta: bool       = Query(False, description="Enable Test-Time Augmentation"),
):
    """Single model prediction (backward compatible)."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image (jpg, png, ...)")

    start     = time.perf_counter()
    img_bytes = await file.read()
    img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    result = predict_image(model, img, use_tta=use_tta)
    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    result["method"] = "single_model"
    return result


@app.post("/predict/ensemble")
async def predict_ensemble(
    file: UploadFile = File(...),
    return_individual: bool = Query(False, description="Return individual model predictions"),
):
    """Ensemble prediction: average of 3 models for better robustness."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image (jpg, png, ...)")

    start     = time.perf_counter()
    img_bytes = await file.read()
    img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Prepare tensor
    from torchvision import transforms as T
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0)

    result = ensemble.predict(tensor, return_individual=return_individual)
    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    result["method"] = "ensemble"
    return result


@app.post("/predict/robust")
async def predict_robust_tta(
    file: UploadFile = File(...),
):
    """Robust TTA: single model with advanced augmentations (noise, blur, lighting)."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image (jpg, png, ...)")

    start     = time.perf_counter()
    img_bytes = await file.read()
    img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    result = predict_with_robust_tta(model, img, device=DEVICE)
    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    result["method"] = "single_model+robust_tta"
    return result


@app.post("/predict/ultra")
async def predict_ultra_robust(
    file: UploadFile = File(...),
):
    """Ultra-robust: Ensemble + Robust TTA (most reliable for production)."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image (jpg, png, ...)")

    start     = time.perf_counter()
    img_bytes = await file.read()
    img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    result = ensemble_predict_with_tta(ensemble, img, device=DEVICE)
    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return result
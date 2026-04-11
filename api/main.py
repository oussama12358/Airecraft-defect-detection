import time, io
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from api.inference import load_model, predict_image
from api.schemas import PredictionResponse

app = FastAPI(
    title="Aircraft Defect Detection API",
    description="Classifies surface defects using EfficientNet-B3 + Grad-CAM",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

model = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@app.on_event("startup")
async def startup():
    global model
    model = load_model("checkpoints/best_model.pt")
    model.to(DEVICE)
    print(f"[API] Model loaded on {DEVICE}")


@app.get("/")
def root():
    return JSONResponse({
        "message": "Aircraft Defect Detection API",
        "status_url": "/health",
        "predict_url": "/predict",
        "docs_url": "/docs",
    })


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/health")
def health():
    return {"status": "healthy", "device": DEVICE, "model": "efficientnet_b3"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file:    UploadFile = File(...),
    use_tta: bool       = Query(False, description="Enable Test-Time Augmentation"),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image (jpg, png, ...)")

    start     = time.perf_counter()
    img_bytes = await file.read()
    img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    result = predict_image(model, img, use_tta=use_tta)
    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return result
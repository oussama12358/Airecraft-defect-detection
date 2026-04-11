from pydantic import BaseModel
from typing import Dict


class PredictionResponse(BaseModel):
    predicted_class:        str
    confidence:             float
    all_probabilities:      Dict[str, float]
    gradcam_heatmap_base64: str
    latency_ms:             float
    tta_used:               bool = False
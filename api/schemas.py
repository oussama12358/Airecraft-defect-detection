from pydantic import BaseModel
from typing import Dict, Optional, Any


class PredictionResponse(BaseModel):
    """Standard prediction response."""
    predicted_class:        str
    confidence:             float
    all_probabilities:      Dict[str, float]
    gradcam_heatmap_base64: str = ""
    latency_ms:             float
    tta_used:               bool = False
    method:                 str = "single_model"
    

class EnsemblePredictionResponse(BaseModel):
    """Ensemble prediction response with individual model outputs."""
    predicted_class:        str
    confidence:             float
    all_probabilities:      Dict[str, float]
    ensemble_size:          int
    individual_models:      Optional[Dict[str, Dict]] = None
    latency_ms:             float
    method:                 str = "ensemble"


class RobustPredictionResponse(BaseModel):
    """Robust TTA response with uncertainty estimates."""
    predicted_class:        str
    confidence:             float
    all_probabilities:      Dict[str, float]
    uncertainty:            Dict[str, float]  # Uncertainty per class
    tta_passes:             int
    latency_ms:             float
    method:                 str = "robust_tta"


class UltraRobustPredictionResponse(BaseModel):
    """Ultimate robustness: Ensemble + Robust TTA."""
    predicted_class:        str
    confidence:             float
    all_probabilities:      Dict[str, float]
    uncertainty:            Dict[str, float]
    ensemble_size:          int
    tta_passes:             int
    latency_ms:             float
    method:                 str = "ensemble+tta"
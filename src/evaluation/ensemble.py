"""
Ensemble inference for improved robustness and reduced overfitting.
Combines multiple models and TTA augmentations for production-grade predictions.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ──────────────────────────────────────────────────────────────────────────────
# ROBUST TTA WITH PERTURBATION SIMULATION
# ──────────────────────────────────────────────────────────────────────────────

def get_robust_tta_transforms() -> List:
    """
    Enhanced TTA with augmentations that simulate real-world conditions:
    - Geometric transforms (flips, rotations)
    - Lighting changes (brightness, contrast)
    - Compression artifacts (slight blur)
    - Noise simulation
    """
    base = [T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)]
    
    transforms_list = [
        # Original
        T.Compose(base),
        
        # Geometric transforms
        T.Compose([T.RandomHorizontalFlip(p=1.0)] + base),
        T.Compose([T.RandomVerticalFlip(p=1.0)] + base),
        T.Compose([T.RandomRotation((90, 90))] + base),
        T.Compose([T.RandomRotation((180, 180))] + base),
        
        # Lighting changes
        T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        ] + base),
        
        # Simulated compression/blur
        T.Compose([
            T.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)),
        ] + base),
        
        # Combined: rotation + lighting
        T.Compose([
            T.RandomRotation((45, 45)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ] + base),
    ]
    
    return transforms_list


@torch.no_grad()
def predict_with_robust_tta(
    model, 
    img: Image.Image, 
    device: str = "cpu",
    num_passes: int = None
) -> dict:
    """
    Robust TTA prediction with uncertainty estimation.
    
    Args:
        model: Trained neural network
        img: PIL Image
        device: 'cpu' or 'cuda'
        num_passes: Override number of TTA passes (default: all transforms)
    
    Returns:
        dict with predictions, confidence, and uncertainty
    """
    model.eval()
    model = model.to(device)
    
    transforms_list = get_robust_tta_transforms()
    if num_passes is not None:
        transforms_list = transforms_list[:num_passes]
    
    all_probs = []
    
    for transform in transforms_list:
        tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        all_probs.append(probs)
    
    all_probs = np.array(all_probs)
    
    # Ensemble predictions
    avg_probs = all_probs.mean(axis=0)
    pred_idx = avg_probs.argmax()
    confidence = float(avg_probs[pred_idx])
    
    # Uncertainty: standard deviation of predictions across augmentations
    uncertainty = all_probs.std(axis=0)
    
    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": round(confidence * 100, 2),
        "all_probabilities": {
            c: round(float(p) * 100, 2)
            for c, p in zip(CLASS_NAMES, avg_probs)
        },
        "uncertainty": {
            c: round(float(u) * 100, 2)
            for c, u in zip(CLASS_NAMES, uncertainty)
        },
        "tta_passes": len(transforms_list),
    }


# ──────────────────────────────────────────────────────────────────────────────
# MODEL ENSEMBLE FOR ROBUSTNESS
# ──────────────────────────────────────────────────────────────────────────────

class ModelEnsemble:
    """
    Ensemble of multiple models for improved generalization and robustness.
    Reduces impact of individual model overfitting.
    """
    
    def __init__(self, models: Dict[str, torch.nn.Module], device: str = "cpu"):
        """
        Args:
            models: dict of {model_name: model_instance}
            device: 'cpu' or 'cuda'
        """
        self.models = {name: m.to(device).eval() for name, m in models.items()}
        self.device = device
        
    @torch.no_grad()
    def predict(self, tensor: torch.Tensor, return_individual: bool = False) -> dict:
        """
        Ensemble prediction by averaging logits across models.
        
        Args:
            tensor: input tensor (B, C, H, W)
            return_individual: if True, return individual model predictions too
        
        Returns:
            dict with ensemble prediction and optional individual predictions
        """
        tensor = tensor.to(self.device)
        all_logits = []
        individual_preds = {}
        
        for model_name, model in self.models.items():
            logits = model(tensor)
            all_logits.append(logits)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            individual_preds[model_name] = {
                "class": CLASS_NAMES[probs.argmax()],
                "confidence": round(float(probs.max()) * 100, 2),
            }
        
        # Ensemble: average logits then softmax
        avg_logits = torch.stack(all_logits).mean(dim=0)
        probs = F.softmax(avg_logits, dim=1)[0].cpu().numpy()
        pred_idx = probs.argmax()
        confidence = float(probs[pred_idx])
        
        result = {
            "predicted_class": CLASS_NAMES[pred_idx],
            "confidence": round(confidence * 100, 2),
            "all_probabilities": {
                c: round(float(p) * 100, 2)
                for c, p in zip(CLASS_NAMES, probs)
            },
            "ensemble_size": len(self.models),
        }
        
        if return_individual:
            result["individual_models"] = individual_preds
        
        return result


@torch.no_grad()
def ensemble_predict_with_tta(
    ensemble: ModelEnsemble,
    img: Image.Image,
    device: str = "cpu",
) -> dict:
    """
    Ultimate robustness: Ensemble + Robust TTA.
    
    This combines:
    1. Multiple models (reduce individual model bias)
    2. Multiple augmentations (reduce positional/lighting bias)
    
    Result: Very robust but slower predictions.
    """
    ensemble.models = {k: v.to(device) for k, v in ensemble.models.items()}
    
    transforms_list = get_robust_tta_transforms()
    all_probs = []
    
    # For each augmentation
    for transform in transforms_list:
        tensor = transform(img.convert("RGB")).unsqueeze(0)
        
        # Average across models
        model_probs = []
        for model in ensemble.models.values():
            logits = model(tensor.to(device))
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            model_probs.append(probs)
        
        avg_probs = np.array(model_probs).mean(axis=0)
        all_probs.append(avg_probs)
    
    # Average across augmentations and models
    final_probs = np.array(all_probs).mean(axis=0)
    pred_idx = final_probs.argmax()
    confidence = float(final_probs[pred_idx])
    
    # Uncertainty: variance across all (model, augmentation) pairs
    all_probs_flat = np.array(all_probs).flatten().reshape(-1, len(CLASS_NAMES))
    uncertainty = all_probs_flat.std(axis=0)
    
    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": round(confidence * 100, 2),
        "all_probabilities": {
            c: round(float(p) * 100, 2)
            for c, p in zip(CLASS_NAMES, final_probs)
        },
        "uncertainty": {
            c: round(float(u) * 100, 2)
            for c, u in zip(CLASS_NAMES, uncertainty)
        },
        "ensemble_size": len(ensemble.models),
        "tta_passes": len(transforms_list),
        "method": "ensemble+tta",
    }

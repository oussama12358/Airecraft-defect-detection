import io, base64, torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from src.models.efficientnet_b3 import build_efficientnet_b3
from src.explainability.gradcam import GradCAM
from src.evaluation.tta import predict_with_tta
from omegaconf import OmegaConf

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

cfg = OmegaConf.to_container(OmegaConf.load("configs/config.yaml"), resolve=True)


def load_model(checkpoint_path: str, num_classes: int = 6):
    model = build_efficientnet_b3(num_classes)
    if cfg["training"].get("use_lora", False):
        from src.training.lora import apply_lora
        model = apply_lora(model, r=cfg["training"]["lora_rank"], alpha=cfg["training"]["lora_alpha"], dropout=cfg["training"]["lora_dropout"], target_modules=cfg["training"].get("lora_target_modules", ["fc", "classifier"]))
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def predict_image(model, img: Image.Image, use_tta: bool = False) -> dict:
    # ── TTA path ──────────────────────────────────────────────────────────────
    if use_tta:
        result = predict_with_tta(model, img)
        result["tta_used"] = True
        result["gradcam_heatmap_base64"] = ""   # TTA skips Grad-CAM
        return result

    # ── Standard path ─────────────────────────────────────────────────────────
    tensor = TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()

    # Grad-CAM
    gradcam = GradCAM(model, model.features[-1])
    cam, _  = gradcam.generate(tensor, pred_idx)
    overlay = gradcam.overlay(np.array(img.resize((224, 224))), cam)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "predicted_class":        CLASS_NAMES[pred_idx],
        "confidence":             round(float(probs[pred_idx]) * 100, 2),
        "all_probabilities":      {
            c: round(float(p) * 100, 2)
            for c, p in zip(CLASS_NAMES, probs)
        },
        "gradcam_heatmap_base64": heatmap_b64,
        "tta_used":               False,
    }
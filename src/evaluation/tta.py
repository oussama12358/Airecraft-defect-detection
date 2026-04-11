import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_tta_transforms():
    base = [T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)]
    return [
        T.Compose(base),
        T.Compose([T.RandomHorizontalFlip(p=1.0)] + base),
        T.Compose([T.RandomVerticalFlip(p=1.0)]   + base),
        T.Compose([T.RandomRotation((90, 90))]    + base),
        T.Compose([T.RandomRotation((180, 180))]  + base),
    ]


@torch.no_grad()
def predict_with_tta(model, img: Image.Image, device: str = "cpu") -> dict:
    model.eval()
    all_probs = []

    for transform in get_tta_transforms():
        tensor = transform(img).unsqueeze(0).to(device)
        probs  = F.softmax(model(tensor), dim=1).squeeze(0)
        all_probs.append(probs)

    avg_probs = torch.stack(all_probs).mean(0)
    pred_idx  = avg_probs.argmax().item()

    return {
        "predicted_class":   CLASS_NAMES[pred_idx],
        "confidence":        round(float(avg_probs[pred_idx]) * 100, 2),
        "all_probabilities": {
            c: round(float(p) * 100, 2)
            for c, p in zip(CLASS_NAMES, avg_probs)
        },
        "tta_passes": len(get_tta_transforms()),
    }
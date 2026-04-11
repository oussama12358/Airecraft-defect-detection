import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]


@torch.no_grad()
def evaluate_model(model, loader, device: str, reports_dir: str = "reports", model_name: str = "model"):
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    model.eval()

    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    # ── Classification report ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cmap="Blues",
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = f"{reports_dir}/confusion_matrix_{model_name}.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[Metrics] Confusion matrix saved → {cm_path}")

    return cm
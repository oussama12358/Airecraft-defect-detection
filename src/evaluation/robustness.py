import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader
from pathlib import Path

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]


# ── Perturbation functions ────────────────────────────────────────────────────

def add_gaussian_noise(img: torch.Tensor, std: float) -> torch.Tensor:
    """Add Gaussian noise to a normalized tensor."""
    return (img + torch.randn_like(img) * std).clamp(0, 1)


def add_blur(img: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Apply Gaussian blur."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return T.GaussianBlur(kernel_size=k, sigma=(0.5, 2.0))(img)


def change_brightness(img: torch.Tensor, factor: float) -> torch.Tensor:
    """Adjust brightness."""
    return TF.adjust_brightness(img, brightness_factor=factor).clamp(0, 1)


def add_jpeg_compression(img: torch.Tensor, quality: int) -> torch.Tensor:
    """Simulate JPEG compression artifacts."""
    import io
    pil = TF.to_pil_image(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return TF.to_tensor(Image.open(buf))


# ── Perturbation configs ──────────────────────────────────────────────────────

PERTURBATIONS = {
    "gaussian_noise_low":    lambda x: add_gaussian_noise(x, std=0.05),
    "gaussian_noise_medium": lambda x: add_gaussian_noise(x, std=0.15),
    "gaussian_noise_high":   lambda x: add_gaussian_noise(x, std=0.30),
    "blur_light":            lambda x: add_blur(x, kernel_size=3),
    "blur_heavy":            lambda x: add_blur(x, kernel_size=9),
    "brightness_dark":       lambda x: change_brightness(x, factor=0.4),
    "brightness_bright":     lambda x: change_brightness(x, factor=1.8),
    "jpeg_quality_50":       lambda x: add_jpeg_compression(x, quality=50),
    "jpeg_quality_20":       lambda x: add_jpeg_compression(x, quality=20),
}


# ── Core evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_with_perturbation(model, loader, device, perturbation_fn=None):
    """Evaluate model accuracy with optional perturbation applied."""
    model.eval()
    correct, total = 0, 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        if perturbation_fn is not None:
            imgs = torch.stack([perturbation_fn(img.cpu()) for img in imgs]).to(device)
        preds = model(imgs).argmax(1)
        correct += (preds == labels.to(device)).sum().item()
        total   += labels.size(0)

    return correct / total


def run_robustness_evaluation(
    model,
    loader,
    device:      str  = "cpu",
    reports_dir: str  = "reports",
) -> pd.DataFrame:
    """
    Run full robustness evaluation across all perturbations.
    Returns a DataFrame with results and saves plots.
    """
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    print("[Robustness] Evaluating clean accuracy...")
    clean_acc = evaluate_with_perturbation(model, loader, device)
    print(f"  Clean accuracy: {clean_acc*100:.2f}%")

    results = [{"perturbation": "clean", "accuracy": clean_acc, "drop": 0.0}]

    for name, fn in PERTURBATIONS.items():
        print(f"[Robustness] Testing: {name}...")
        acc  = evaluate_with_perturbation(model, loader, device, fn)
        drop = clean_acc - acc
        results.append({
            "perturbation": name,
            "accuracy":     acc,
            "drop":         drop,
        })
        print(f"  Accuracy: {acc*100:.2f}%  |  Drop: {drop*100:.2f}%")

    df = pd.DataFrame(results)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = f"{reports_dir}/robustness_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Robustness] Results saved → {csv_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot_robustness(df, reports_dir)

    return df


def _plot_robustness(df: pd.DataFrame, reports_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = ["green" if r["perturbation"] == "clean"
              else "red" if r["drop"] > 0.10
              else "orange" if r["drop"] > 0.05
              else "steelblue"
              for _, r in df.iterrows()]

    # Accuracy bar chart
    ax1.barh(df["perturbation"], df["accuracy"] * 100, color=colors)
    ax1.axvline(x=df[df["perturbation"] == "clean"]["accuracy"].values[0] * 100,
                color="black", linestyle="--", linewidth=1.5, label="Clean baseline")
    ax1.set_xlabel("Accuracy (%)")
    ax1.set_title("Model Accuracy per Perturbation")
    ax1.legend()
    ax1.set_xlim(0, 105)

    # Drop bar chart
    drop_colors = ["red" if d > 0.10 else "orange" if d > 0.05 else "steelblue"
                   for d in df["drop"]]
    ax2.barh(df["perturbation"], df["drop"] * 100, color=drop_colors)
    ax2.set_xlabel("Accuracy Drop (%)")
    ax2.set_title("Accuracy Drop vs Clean Baseline")
    ax2.axvline(x=5,  color="orange", linestyle="--", linewidth=1, label=">5% warning")
    ax2.axvline(x=10, color="red",    linestyle="--", linewidth=1, label=">10% critical")
    ax2.legend()

    plt.tight_layout()
    path = f"{reports_dir}/robustness_plot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Robustness] Plot saved → {path}")


def compare_models_robustness(
    models:      dict,
    loader,
    device:      str = "cpu",
    reports_dir: str = "reports",
) -> pd.DataFrame:
    """
    Compare robustness across multiple models.
    models = {"ResNet50": model1, "EfficientNet-B3": model2, ...}
    """
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    rows = []

    for model_name, model in models.items():
        print(f"\n[Robustness] Model: {model_name}")
        clean = evaluate_with_perturbation(model, loader, device)
        row   = {"model": model_name, "clean": clean}

        for pert_name, fn in PERTURBATIONS.items():
            acc = evaluate_with_perturbation(model, loader, device, fn)
            row[pert_name] = acc

        rows.append(row)
        print(f"  Clean: {clean*100:.2f}%")

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            "No models were provided for comparison. "
            "Use --resnet_ckpt, --efficientnet_ckpt, or --baseline_ckpt with --mode compare."
        )

    # ── Heatmap ───────────────────────────────────────────────────────────────
    plt.figure(figsize=(14, 5))
    import seaborn as sns

    numeric_cols = ["clean"] + list(PERTURBATIONS.keys())
    heatmap_data = df.set_index("model")[numeric_cols] * 100

    sns.heatmap(
        heatmap_data,
        annot=True, fmt=".1f",
        cmap="RdYlGn",
        vmin=50, vmax=100,
        linewidths=0.5,
    )
    plt.title("Model Robustness Comparison (Accuracy %)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = f"{reports_dir}/robustness_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n[Robustness] Comparison heatmap saved → {path}")

    df.to_csv(f"{reports_dir}/robustness_comparison.csv", index=False)
    return df
import torch
import pandas as pd
from torch.utils.data import WeightedRandomSampler
from collections import Counter

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]


def build_weighted_sampler(csv_path: str) -> WeightedRandomSampler:
    df     = pd.read_csv(csv_path)
    labels = df["label"].tolist()
    counts = Counter(labels)

    print("[Sampler] Class distribution:")
    for cls, n in sorted(counts.items()):
        print(f"  {cls:25s}: {n}")

    class_weights  = {c: 1.0 / counts[c] for c in counts}
    sample_weights = torch.tensor(
        [class_weights[l] for l in labels], dtype=torch.float
    )

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_class_weights_tensor(csv_path: str, device: str = "cpu") -> torch.Tensor:
    df     = pd.read_csv(csv_path)
    counts = df["label"].value_counts()

    weights = torch.tensor(
        [1.0 / counts.get(c, 1) for c in CLASS_NAMES],
        dtype=torch.float,
    )
    weights = weights / weights.sum() * len(CLASS_NAMES)

    print("[Loss Weights] Per-class weights:")
    for cls, w in zip(CLASS_NAMES, weights):
        print(f"  {cls:25s}: {w:.4f}")

    return weights.to(device)
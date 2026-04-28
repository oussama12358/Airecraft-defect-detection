import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


def build_optimizer(model, cfg: dict):
    return optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )


def build_scheduler(optimizer, cfg: dict):
    name = cfg["training"]["scheduler"]
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    elif name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    else:
        raise ValueError(f"Unknown scheduler: {name}")
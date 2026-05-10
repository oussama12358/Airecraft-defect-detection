import argparse

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.datasets.neu_dataset  import NEUDefectDataset
from src.datasets.transforms   import get_transforms
from src.datasets.sampler      import build_weighted_sampler, get_class_weights_tensor
from src.models.baseline_cnn   import BaselineCNN
from src.models.resnet50       import build_resnet50
from src.models.efficientnet_b3   import build_efficientnet_b3
from src.training.trainer      import Trainer
from src.training.scheduler    import build_optimizer, build_scheduler
from src.training.lora         import apply_lora


def parse_args():
    parser = argparse.ArgumentParser(description="Train a defect classifier")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", choices=["baseline_cnn", "resnet50", "efficientnet_b3"])
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--val_csv", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None,
                        help="Checkpoint/report prefix, useful for K-fold runs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--use_ema", action="store_true",
                        help="Enable EMA validation/checkpoint weights for this run")
    parser.add_argument("--ema_decay", type=float, default=None)
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)

    if args.model:
        cfg["model"]["name"] = args.model
    if args.train_csv:
        cfg["data"]["train_csv"] = args.train_csv
    if args.val_csv:
        cfg["data"]["val_csv"] = args.val_csv
    if args.run_name:
        cfg["training"]["run_name"] = args.run_name
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.use_ema:
        cfg["training"]["use_ema"] = True
    if args.ema_decay is not None:
        cfg["training"]["ema_decay"] = args.ema_decay

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Device: {device}")
    print(f"[Train] Train CSV: {cfg['data']['train_csv']}")
    print(f"[Train] Val CSV:   {cfg['data']['val_csv']}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = NEUDefectDataset(cfg["data"]["train_csv"], cfg["data"]["img_dir"],
                                transform=get_transforms("train"))
    val_ds   = NEUDefectDataset(cfg["data"]["val_csv"],   cfg["data"]["img_dir"],
                                transform=get_transforms("val"))

    sampler = build_weighted_sampler(cfg["data"]["train_csv"])

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              sampler=sampler, num_workers=cfg["data"]["num_workers"],
                              pin_memory=(device=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"],
                              shuffle=False, num_workers=cfg["data"]["num_workers"],
                              pin_memory=(device=="cuda"))

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name = cfg["model"]["name"]
    if model_name == "baseline_cnn":
        model = BaselineCNN(cfg["model"]["num_classes"])
    elif model_name == "resnet50":
        model = build_resnet50(cfg["model"]["num_classes"],
                               cfg["model"]["freeze_backbone"])
    else:
        model = build_efficientnet_b3(cfg["model"]["num_classes"],
                                      cfg["model"]["freeze_backbone"])

    if cfg["training"].get("use_lora", False):
        print("[Train] Applying LoRA to model linear layers...")
        model = apply_lora(
            model,
            r=cfg["training"]["lora_rank"],
            alpha=cfg["training"]["lora_alpha"],
            dropout=cfg["training"]["lora_dropout"],
            target_modules=cfg["training"].get("lora_target_modules", ["fc", "classifier"]),
        )

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    class_weights = get_class_weights_tensor(cfg["data"]["train_csv"], device)
    criterion     = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg["training"]["label_smoothing"],
    )
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(model, optimizer, criterion, scheduler, device, cfg)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()

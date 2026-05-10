import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.datasets.neu_dataset  import NEUDefectDataset
from src.datasets.transforms   import get_transforms
from src.models.baseline_cnn   import BaselineCNN
from src.models.resnet50       import build_resnet50
from src.models.efficientnet_b3 import build_efficientnet_b3
from src.evaluation.metrics    import evaluate_model
from src.evaluation.report     import save_report

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]


def infer_model_name(checkpoint: str) -> str:
    stem = Path(checkpoint).stem.replace("best_", "")
    for candidate in ["efficientnet_b3", "baseline_cnn", "resnet50"]:
        if candidate in stem:
            return candidate
    return stem


def build_model(model_name: str, cfg: dict):
    if model_name == "baseline_cnn":
        return BaselineCNN(cfg["model"]["num_classes"])
    if model_name == "resnet50":
        return build_resnet50(cfg["model"]["num_classes"],
                              cfg["model"]["freeze_backbone"])
    if model_name == "efficientnet_b3":
        return build_efficientnet_b3(cfg["model"]["num_classes"],
                                     cfg["model"]["freeze_backbone"])
    raise ValueError(f"Unknown model name: {model_name}")


def main(
    checkpoint: str,
    use_tta: bool = False,
    model_name: str | None = None,
    test_csv: str | None = None,
    report_name: str | None = None,
):
    cfg    = OmegaConf.to_container(OmegaConf.load("configs/config.yaml"), resolve=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = model_name or infer_model_name(checkpoint)
    report_name = report_name or model_name
    test_csv = test_csv or cfg["data"]["test_csv"]
    model = build_model(model_name, cfg)

    if cfg["training"].get("use_lora", False):
        from src.training.lora import apply_lora
        model = apply_lora(
            model,
            r=cfg["training"]["lora_rank"],
            alpha=cfg["training"]["lora_alpha"],
            dropout=cfg["training"]["lora_dropout"],
            target_modules=cfg["training"].get("lora_target_modules", ["fc", "classifier"]),
        )

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)

    print(f"[Eval] Model: {model_name}")
    print(f"[Eval] Test CSV: {test_csv}")
    print("[Eval] Using clean evaluation transforms only.")

    test_ds = NEUDefectDataset(test_csv, cfg["data"]["img_dir"],
                               transform=get_transforms("test"))
    loader  = DataLoader(test_ds, batch_size=32, shuffle=False,
                         num_workers=cfg["data"]["num_workers"])

    metrics = evaluate_model(model, loader, device, cfg["paths"]["reports_dir"], report_name)
    metrics["test_csv"] = test_csv
    save_report(metrics, report_name, cfg["paths"]["reports_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tta",        action="store_true")
    parser.add_argument("--model", choices=["baseline_cnn", "resnet50", "efficientnet_b3"],
                        default=None)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--report_name", type=str, default=None)
    args = parser.parse_args()
    main(args.checkpoint, args.tta, args.model, args.test_csv, args.report_name)

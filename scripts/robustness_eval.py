import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import argparse
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from src.datasets.neu_dataset     import NEUDefectDataset
from src.datasets.transforms      import get_transforms
from src.models.resnet50          import build_resnet50
from src.models.efficientnet_b3      import build_efficientnet_b3
from src.models.baseline_cnn      import BaselineCNN
from src.evaluation.robustness    import run_robustness_evaluation, compare_models_robustness

def load_model(name: str, checkpoint: str, cfg: dict, num_classes: int = 6):
    if name == "resnet50":
        model = build_resnet50(num_classes)
    elif name == "efficientnet_b3":
        model = build_efficientnet_b3(num_classes)
    else:
        model = BaselineCNN(num_classes)

    if cfg["training"].get("use_lora", False):
        from src.training.lora import apply_lora
        model = apply_lora(
            model,
            r=cfg["training"]["lora_rank"],
            alpha=cfg["training"]["lora_alpha"],
            dropout=cfg["training"]["lora_dropout"],
            target_modules=cfg["training"].get("lora_target_modules", ["fc", "classifier"]),
        )

    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "compare"],
                        default="single", help="single model or compare all")
    parser.add_argument("--checkpoint",      type=str, default=None,
                        help="Path to .pt checkpoint (for single mode)")
    parser.add_argument("--model",           type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b3", "baseline_cnn"])
    parser.add_argument("--resnet_ckpt",     type=str, default=None)
    parser.add_argument("--efficientnet_ckpt", type=str, default=None)
    parser.add_argument("--baseline_ckpt",   type=str, default=None)
    args = parser.parse_args()

    if args.mode == "single" and args.checkpoint is None:
        parser.error("--checkpoint is required for single mode")

    cfg    = OmegaConf.to_container(OmegaConf.load("configs/config.yaml"), resolve=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_ds = NEUDefectDataset(
        cfg["data"]["test_csv"],
        cfg["data"]["img_dir"],
        transform=get_transforms("val"),
    )
    loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                        num_workers=cfg["data"]["num_workers"])

    if args.mode == "single":
        model = load_model(args.model, args.checkpoint, cfg)
        model.to(device)
        run_robustness_evaluation(model, loader, device,
                                  cfg["paths"]["reports_dir"])

    else:
        # Compare all 3 models
        if not (args.resnet_ckpt or args.efficientnet_ckpt or args.baseline_ckpt):
            parser.error(
                "--resnet_ckpt, --efficientnet_ckpt, or --baseline_ckpt is required for compare mode"
            )

        models = {}
        if args.resnet_ckpt:
            models["ResNet50"] = load_model("resnet50", args.resnet_ckpt, cfg).to(device)
        if args.efficientnet_ckpt:
            models["EfficientNet-B3"] = load_model("efficientnet_b3",
                                                    args.efficientnet_ckpt, cfg).to(device)
        if args.baseline_ckpt:
            models["Baseline CNN"] = load_model("baseline_cnn",
                                                 args.baseline_ckpt, cfg).to(device)

        compare_models_robustness(models, loader, device,
                                  cfg["paths"]["reports_dir"])


if __name__ == "__main__":
    main()
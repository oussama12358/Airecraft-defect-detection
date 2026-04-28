import torch, argparse
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.datasets.neu_dataset  import NEUDefectDataset
from src.datasets.transforms   import get_transforms
from src.models.baseline_cnn   import BaselineCNN
from src.models.resnet50       import build_resnet50
from src.models.efficientnet_b3 import build_efficientnet_b3
from src.evaluation.metrics    import evaluate_model
from src.evaluation.report     import save_report
from sklearn.metrics           import classification_report
import numpy as np

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]


def main(checkpoint: str, use_tta: bool = False):
    cfg    = OmegaConf.to_container(OmegaConf.load("configs/config.yaml"), resolve=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Extract model name from checkpoint path (e.g., "checkpoints/best_resnet50.pt" -> "resnet50")
    checkpoint_name = checkpoint.split("/")[-1].replace("best_", "").replace(".pt", "")
    model_name = checkpoint_name
    
    if model_name == "baseline_cnn":
        model = BaselineCNN(cfg["model"]["num_classes"])
    elif model_name == "resnet50":
        model = build_resnet50(cfg["model"]["num_classes"],
                               cfg["model"]["freeze_backbone"])
    else:
        model = build_efficientnet_b3(cfg["model"]["num_classes"],
                                      cfg["model"]["freeze_backbone"])

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

    test_ds = NEUDefectDataset(cfg["data"]["test_csv"], cfg["data"]["img_dir"],
                               transform=get_transforms("val"))
    loader  = DataLoader(test_ds, batch_size=32, shuffle=False,
                         num_workers=cfg["data"]["num_workers"])

    cm = evaluate_model(model, loader, device, cfg["paths"]["reports_dir"], model_name)
    save_report({"confusion_matrix": cm.tolist()}, model_name,
                cfg["paths"]["reports_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tta",        action="store_true")
    args = parser.parse_args()
    main(args.checkpoint, args.tta)
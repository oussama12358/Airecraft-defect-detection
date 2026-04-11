import torch.nn as nn
from torchvision import models


def build_efficientnet_b3(num_classes: int = 6, freeze_backbone: bool = True) -> nn.Module:
    model = models.efficientnet_b3(
        weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
    )

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in list(model.features.children())[-3:]:
            for p in param.parameters():
                p.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model
import torch.nn as nn


class BaselineCNN(nn.Module):
    """Simple 5-block CNN — used as benchmark baseline."""

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3,   32),
            self._block(32,  64),
            self._block(64,  128),
            self._block(128, 256),
            self._block(256, 512),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
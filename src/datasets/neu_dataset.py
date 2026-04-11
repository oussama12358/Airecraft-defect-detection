import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]


class NEUDefectDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.img_dir   = img_dir
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(
            os.path.join(self.img_dir, row["filename"])
        ).convert("RGB")
        label = self.class_to_idx[row["label"]]
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_labels(self):
        """Returns all labels as integers (used by sampler)."""
        return [self.class_to_idx[l] for l in self.df["label"]]
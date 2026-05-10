import torchvision.transforms as T

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str):
    split = split.lower()

    if split == "train":
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.RandomErasing(p=0.2),
            T.Normalize(MEAN, STD),
        ])

    if split in {"val", "validation", "test", "eval", "clean"}:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])

    raise ValueError(
        f"Unknown split '{split}'. Use 'train' for augmentation or "
        "'val'/'test' for clean evaluation."
    )

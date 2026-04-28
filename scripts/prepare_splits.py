import os
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]


def prepare_splits(
    raw_dir:       str   = "data/raw",
    processed_dir: str   = "data/processed/images",
    splits_dir:    str   = "data/splits",
    train_ratio:   float = 0.70,
    val_ratio:     float = 0.15,
    img_size:      int   = 224,
):
    from PIL import Image

    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    Path(splits_dir).mkdir(parents=True, exist_ok=True)

    records = []

    # ── Find images ──────────────────────────────────────────────────────────
    # Support both flat layout and train/ subfolder layout from Kaggle
    for class_name in CLASS_NAMES:
        candidates = [
            Path(raw_dir) / class_name,
            Path(raw_dir) / "train" / class_name,
            Path(raw_dir) / "train" / "images" / class_name,
            Path(raw_dir) / "NEU-DET" / class_name,
            Path(raw_dir) / "NEU-DET" / "train" / "images" / class_name,
        ]
        class_dir = next((p for p in candidates if p.exists()), None)

        if class_dir is None:
            print(f"[WARNING] Folder not found for class: {class_name}")
            continue

        img_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.bmp"))
        print(f"[Prepare] {class_name}: {len(img_files)} images found")

        for img_path in img_files:
            dest = Path(processed_dir) / f"{class_name}_{img_path.name}"
            if not dest.exists():
                img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
                img.save(dest)

            records.append({
                "filename": dest.name,
                "label":    class_name,
            })

    if not records:
        raise RuntimeError("No images found! Check your raw_dir path.")

    df = pd.DataFrame(records)
    print(f"\n[Prepare] Total images: {len(df)}")

    # ── Stratified splits ─────────────────────────────────────────────────────
    test_ratio = 1.0 - train_ratio - val_ratio

    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio),
        stratify=df["label"], random_state=42,
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=(test_ratio / (val_ratio + test_ratio)),
        stratify=temp_df["label"], random_state=42,
    )

    train_df.to_csv(f"{splits_dir}/train.csv", index=False)
    val_df.to_csv(f"{splits_dir}/val.csv",   index=False)
    test_df.to_csv(f"{splits_dir}/test.csv",  index=False)

    print(f"[Splits] train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")
    print("[Splits] CSV files saved to data/splits/")


if __name__ == "__main__":
    prepare_splits()
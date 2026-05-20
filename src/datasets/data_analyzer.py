import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
import torch


class DataAnalyzer:
    """Analyze dataset balance, annotation quality, and augmentations."""
    
    def __init__(self, csv_path: str, img_dir: str):
        """
        Args:
            csv_path: path to split CSV (train/val/test)
            img_dir: path to image directory
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.class_names = [
            "crazing", "inclusion", "patches",
            "pitted_surface", "rolled-in_scale", "scratches"
        ]
        
    def check_data_balance(self) -> dict:
        """Check class distribution balance."""
        if "class" not in self.df.columns and "label" not in self.df.columns:
            print("Warning: No class/label column found. Using filename parsing.")
            self.df["class"] = self.df["filename"].str.split("_").str[0]
        
        class_col = "class" if "class" in self.df.columns else "label"
        counts = self.df[class_col].value_counts()
        
        print("\nData Balance:")
        print(counts)
        print(f"\nTotal samples: {len(self.df)}")
        print(f"Imbalance ratio: {counts.max() / counts.min():.2f}x")
        
        return counts.to_dict()
    
    def check_annotation_quality(self) -> dict:
        """
        Check for potential annotation issues:
        - Missing files
        - Suspicious class labels
        - Consistency
        """
        print("\nAnnotation Quality Check:")
        
        issues = {
            "missing_files": [],
            "unreadable_images": [],
            "invalid_classes": [],
            "duplicates": [],
            "filename_label_mismatch": [],
            "small_images": [],
        }
        
        class_col = "class" if "class" in self.df.columns else "label"
        
        for idx, row in self.df.iterrows():
            filename = row["filename"]
            img_path = os.path.join(self.img_dir, filename)
            
            # Check if file exists
            if not os.path.exists(img_path):
                issues["missing_files"].append(filename)
                continue

            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    img.verify()
                if width < 32 or height < 32:
                    issues["small_images"].append((filename, width, height))
            except Exception as exc:
                issues["unreadable_images"].append((filename, str(exc)))
                continue
            
            # Check if class is valid
            if class_col in self.df.columns:
                cls = row[class_col]
                if cls not in self.class_names:
                    issues["invalid_classes"].append((filename, cls))
                elif not str(filename).startswith(f"{cls}_"):
                    issues["filename_label_mismatch"].append((filename, cls))
        
        # Check duplicates
        duplicates = self.df[self.df.duplicated(subset=["filename"], keep=False)]
        if len(duplicates) > 0:
            issues["duplicates"] = duplicates["filename"].tolist()
        
        print(f"Missing files: {len(issues['missing_files'])}")
        if issues["missing_files"]:
            print(f"   Examples: {issues['missing_files'][:3]}")

        print(f"Unreadable images: {len(issues['unreadable_images'])}")
        if issues["unreadable_images"]:
            print(f"   Examples: {issues['unreadable_images'][:3]}")
        
        print(f"Invalid classes: {len(issues['invalid_classes'])}")
        if issues["invalid_classes"]:
            print(f"   Examples: {issues['invalid_classes'][:3]}")
        
        print(f"Duplicates: {len(issues['duplicates'])}")
        
        print(f"Filename/label mismatches: {len(issues['filename_label_mismatch'])}")
        if issues["filename_label_mismatch"]:
            print(f"   Examples: {issues['filename_label_mismatch'][:3]}")

        print(f"Very small images: {len(issues['small_images'])}")

        return issues
    
    def plot_data_balance(self, output_dir: str = "reports", output_name: str = "data_balance.png"):
        """Visualize class distribution."""
        os.makedirs(output_dir, exist_ok=True)
        
        class_col = "class" if "class" in self.df.columns else "label"
        counts = self.df[class_col].value_counts()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart
        counts.plot(kind="bar", ax=ax1, color="steelblue")
        ax1.set_title("Class Distribution")
        ax1.set_ylabel("Count")
        ax1.set_xlabel("Class")
        ax1.tick_params(axis='x', rotation=45)
        
        # Pie chart
        counts.plot(kind="pie", ax=ax2, autopct="%1.1f%%")
        ax2.set_title("Class Proportions")
        ax2.set_ylabel("")
        
        plt.tight_layout()
        path = os.path.join(output_dir, output_name)
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
        plt.close()


class AugmentationVisualizer:
    """Visualize data augmentations to verify they're appropriate."""
    
    def __init__(self, img_dir: str):
        self.img_dir = img_dir
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def get_augmentation_transforms(self):
        """Define augmentations matching training."""
        return {
            "Original": T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
            ]),
            
            "ColorJitter": T.Compose([
                T.Resize((224, 224)),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
            ]),
            
            "RandomRotation": T.Compose([
                T.Resize((224, 224)),
                T.RandomRotation(15),
                T.ToTensor(),
            ]),
            
            "GaussianBlur": T.Compose([
                T.Resize((224, 224)),
                T.GaussianBlur(kernel_size=3, sigma=(0.5, 2.0)),
                T.ToTensor(),
            ]),
            
            "RandomHFlip": T.Compose([
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(p=1.0),
                T.ToTensor(),
            ]),
            
            "Combined": T.Compose([
                T.Resize((224, 224)),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.3),
                T.ToTensor(),
            ]),
        }

    def _to_display_array(self, img):
        if isinstance(img, torch.Tensor):
            return img.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
        return np.array(img) / 255.0
    
    def visualize_augmentations(self, img_path: str, output_dir: str = "reports"):
        """Show original + augmented versions side by side."""
        os.makedirs(output_dir, exist_ok=True)
        
        img = Image.open(img_path).convert("RGB")
        transforms = self.get_augmentation_transforms()
        
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        for idx, (aug_name, transform) in enumerate(transforms.items()):
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            
            # Apply transform
            aug_img = transform(img)
            
            aug_img = self._to_display_array(aug_img)
            
            ax.imshow(aug_img)
            ax.set_title(aug_name, fontsize=12, fontweight="bold")
            ax.axis("off")
        
        plt.suptitle(f"Data Augmentations - {os.path.basename(img_path)}", fontsize=14, fontweight="bold")
        
        output_path = os.path.join(output_dir, f"augmentations_{Path(img_path).stem}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()
    
    def visualize_batch(self, img_paths: list, output_dir: str = "reports"):
        """Visualize multiple images with augmentations."""
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(18, 12))
        
        for batch_idx, img_path in enumerate(img_paths[:6]):  # Max 6 images
            if not os.path.exists(img_path):
                continue
                
            img = Image.open(img_path).convert("RGB")
            transforms = list(self.get_augmentation_transforms().values())[:3]  # 3 augmentations
            
            for aug_idx, transform in enumerate(transforms):
                ax = plt.subplot(6, 3, batch_idx * 3 + aug_idx + 1)
                
                aug_img = transform(img)
                aug_img = self._to_display_array(aug_img)
                
                ax.imshow(aug_img)
                ax.axis("off")
        
        plt.suptitle("Batch Augmentation Visualization", fontsize=14, fontweight="bold")
        output_path = os.path.join(output_dir, "augmentations_batch.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()

    def visualize_jpeg_zoom(self, img_path: str, output_dir: str = "reports"):
        """Save zoomed crops to inspect JPEG/compression artifacts."""
        os.makedirs(output_dir, exist_ok=True)

        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        crop_size = max(24, min(width, height) // 4)
        center = img.crop((
            (width - crop_size) // 2,
            (height - crop_size) // 2,
            (width + crop_size) // 2,
            (height + crop_size) // 2,
        )).resize((224, 224), Image.Resampling.NEAREST)
        corner = img.crop((0, 0, crop_size, crop_size)).resize(
            (224, 224),
            Image.Resampling.NEAREST,
        )

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, title, sample in zip(
            axes,
            ["Original", "Center zoom", "Corner zoom"],
            [img, center, corner],
        ):
            ax.imshow(sample)
            ax.set_title(title)
            ax.axis("off")

        plt.suptitle(f"JPEG artifact inspection - {os.path.basename(img_path)}")
        output_path = os.path.join(output_dir, f"jpeg_zoom_{Path(img_path).stem}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()


if __name__ == "__main__":
    import torch
    
    # Example usage
    print("=" * 60)
    print("DATA ANALYSIS & AUGMENTATION VISUALIZATION")
    print("=" * 60)
    
    # Analyze training data
    analyzer = DataAnalyzer(
        csv_path="data/splits/train.csv",
        img_dir="data/processed/images"
    )
    
    print("\n1. CHECK DATA BALANCE:")
    analyzer.check_data_balance()
    analyzer.plot_data_balance()
    
    print("\n2. CHECK ANNOTATION QUALITY:")
    analyzer.check_annotation_quality()
    
    # Visualize augmentations
    print("\n3. VISUALIZE AUGMENTATIONS:")
    visualizer = AugmentationVisualizer(img_dir="data/processed/images")
    
    # Get sample images from each class
    from pathlib import Path
    img_files = list(Path("data/processed/images").glob("*.jpg"))[:3]
    
    if img_files:
        for img_path in img_files:
            visualizer.visualize_augmentations(str(img_path))
        visualizer.visualize_batch([str(p) for p in img_files])
    
    print("\nAnalysis complete!")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import Counter
import cv2
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
            print("⚠️  No class/label column found. Using filename parsing.")
            self.df["class"] = self.df["filename"].str.split("_").str[0]
        
        class_col = "class" if "class" in self.df.columns else "label"
        counts = self.df[class_col].value_counts()
        
        print("\n📊 Data Balance:")
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
        print("\n🔍 Annotation Quality Check:")
        
        issues = {
            "missing_files": [],
            "invalid_classes": [],
            "duplicates": [],
        }
        
        class_col = "class" if "class" in self.df.columns else "label"
        
        for idx, row in self.df.iterrows():
            filename = row["filename"]
            img_path = os.path.join(self.img_dir, filename)
            
            # Check if file exists
            if not os.path.exists(img_path):
                issues["missing_files"].append(filename)
            
            # Check if class is valid
            if class_col in self.df.columns:
                cls = row[class_col]
                if cls not in self.class_names:
                    issues["invalid_classes"].append((filename, cls))
        
        # Check duplicates
        duplicates = self.df[self.df.duplicated(subset=["filename"], keep=False)]
        if len(duplicates) > 0:
            issues["duplicates"] = duplicates["filename"].tolist()
        
        print(f"✅ Missing files: {len(issues['missing_files'])}")
        if issues["missing_files"]:
            print(f"   Examples: {issues['missing_files'][:3]}")
        
        print(f"✅ Invalid classes: {len(issues['invalid_classes'])}")
        if issues["invalid_classes"]:
            print(f"   Examples: {issues['invalid_classes'][:3]}")
        
        print(f"✅ Duplicates: {len(issues['duplicates'])}")
        
        return issues
    
    def plot_data_balance(self, output_dir: str = "reports"):
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
        path = os.path.join(output_dir, "data_balance.png")
        plt.savefig(path, dpi=150)
        print(f"✅ Saved: {path}")
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
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                T.ToTensor(),
            ]),
            
            "RandomRotation": T.Compose([
                T.Resize((224, 224)),
                T.RandomRotation(45),
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
                T.RandomRotation(30),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
            ]),
        }
    
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
            
            # Denormalize for display (if it's a tensor)
            if isinstance(aug_img, torch.Tensor):
                aug_img = aug_img.numpy().transpose(1, 2, 0)
                aug_img = (aug_img * np.array(self.std) + np.array(self.mean)).clip(0, 1)
            else:
                aug_img = np.array(aug_img) / 255.0
            
            ax.imshow(aug_img)
            ax.set_title(aug_name, fontsize=12, fontweight="bold")
            ax.axis("off")
        
        plt.suptitle(f"Data Augmentations - {os.path.basename(img_path)}", fontsize=14, fontweight="bold")
        
        output_path = os.path.join(output_dir, f"augmentations_{Path(img_path).stem}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def visualize_batch(self, img_paths: list, output_dir: str = "reports"):
        """Visualize multiple images with augmentations."""
        os.makedirs(output_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(18, 12))
        
        for batch_idx, img_path in enumerate(img_paths[:6]):  # Max 6 images
            if not os.path.exists(img_path):
                continue
                
            img = Image.open(img_path).convert("RGB")
            transforms = list(self.get_augmentation_transforms().values())[:3]  # 3 augmentations
            
            for aug_idx, transform in enumerate(transforms):
                ax = plt.subplot(6, 3, batch_idx * 3 + aug_idx + 1)
                
                aug_img = transform(img)
                if isinstance(aug_img, torch.Tensor):
                    aug_img = aug_img.numpy().transpose(1, 2, 0)
                    aug_img = (aug_img * np.array(self.std) + np.array(self.mean)).clip(0, 1)
                
                ax.imshow(aug_img)
                ax.axis("off")
        
        plt.suptitle("Batch Augmentation Visualization", fontsize=14, fontweight="bold")
        output_path = os.path.join(output_dir, "augmentations_batch.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {output_path}")
        plt.close()


if __name__ == "__main__":
    import torch
    
    # Example usage
    print("=" * 60)
    print("📊 DATA ANALYSIS & AUGMENTATION VISUALIZATION")
    print("=" * 60)
    
    # Analyze training data
    analyzer = DataAnalyzer(
        csv_path="data/splits/train.csv",
        img_dir="data/processed/images"
    )
    
    print("\n1️⃣ CHECK DATA BALANCE:")
    analyzer.check_data_balance()
    analyzer.plot_data_balance()
    
    print("\n2️⃣ CHECK ANNOTATION QUALITY:")
    analyzer.check_annotation_quality()
    
    # Visualize augmentations
    print("\n3️⃣ VISUALIZE AUGMENTATIONS:")
    visualizer = AugmentationVisualizer(img_dir="data/processed/images")
    
    # Get sample images from each class
    from pathlib import Path
    img_files = list(Path("data/processed/images").glob("*.jpg"))[:3]
    
    if img_files:
        for img_path in img_files:
            visualizer.visualize_augmentations(str(img_path))
        visualizer.visualize_batch([str(p) for p in img_files])
    
    print("\n✅ Analysis complete!")

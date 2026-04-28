import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.data_analyzer import DataAnalyzer, AugmentationVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dataset balance and visualize augmentations"
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Which split to analyze"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--visualize_augmentations",
        action="store_true",
        help="Create augmentation visualizations"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=6,
        help="Number of samples to visualize"
    )
    
    args = parser.parse_args()
    
    csv_path = f"data/splits/{args.split}.csv"
    img_dir = "data/processed/images"
    
    print("\n" + "="*70)
    print(f"📊 DATA ANALYSIS - {args.split.upper()} SPLIT")
    print("="*70)
    
    # Initialize analyzer
    analyzer = DataAnalyzer(csv_path, img_dir)
    
    # Step 1: Check balance
    print("\n1️⃣  CHECKING DATA BALANCE...")
    balance = analyzer.check_data_balance()
    analyzer.plot_data_balance(output_dir=args.output_dir)
    
    # Step 2: Check annotation quality
    print("\n2️⃣  CHECKING ANNOTATION QUALITY...")
    issues = analyzer.check_annotation_quality()
    
    # Step 3: Visualize augmentations
    if args.visualize_augmentations:
        print(f"\n3️⃣  VISUALIZING AUGMENTATIONS ({args.sample_size} samples)...")
        visualizer = AugmentationVisualizer(img_dir)
        
        # Get sample images
        df = analyzer.df
        sample_files = df.sample(min(args.sample_size, len(df)))["filename"].tolist()
        
        for i, filename in enumerate(sample_files, 1):
            img_path = Path(img_dir) / filename
            if img_path.exists():
                print(f"   [{i}/{len(sample_files)}] {filename}")
                visualizer.visualize_augmentations(str(img_path), output_dir=args.output_dir)
        
        # Batch visualization
        if sample_files:
            print(f"   Creating batch visualization...")
            img_paths = [str(Path(img_dir) / f) for f in sample_files if (Path(img_dir) / f).exists()]
            if img_paths:
                visualizer.visualize_batch(img_paths, output_dir=args.output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    print(f"✅ Split: {args.split}")
    print(f"✅ Total samples: {len(analyzer.df)}")
    print(f"✅ Missing files: {len(issues['missing_files'])}")
    print(f"✅ Invalid classes: {len(issues['invalid_classes'])}")
    print(f"✅ Duplicates: {len(issues['duplicates'])}")
    print(f"✅ Output saved to: {args.output_dir}")
    print("="*70 + "\n")
    
    # Recommendations
    if issues['missing_files'] or issues['invalid_classes'] or issues['duplicates']:
        print("⚠️  RECOMMENDATIONS:")
        if issues['missing_files']:
            print(f"   - Remove {len(issues['missing_files'])} rows with missing files")
        if issues['invalid_classes']:
            print(f"   - Check {len(issues['invalid_classes'])} rows with invalid classes")
        if issues['duplicates']:
            print(f"   - Remove {len(issues['duplicates'])} duplicate entries")
    else:
        print("✅ DATA LOOKS CLEAN!")
    
    print()


if __name__ == "__main__":
    main()

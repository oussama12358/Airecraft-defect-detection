import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.data_analyzer import DataAnalyzer, AugmentationVisualizer


def analyze_split(split: str, args) -> None:
    csv_path = f"data/splits/{split}.csv"
    img_dir = "data/processed/images"

    print("\n" + "=" * 70)
    print(f"DATA ANALYSIS - {split.upper()} SPLIT")
    print("=" * 70)

    analyzer = DataAnalyzer(csv_path, img_dir)

    print("\n1. CHECKING DATA BALANCE...")
    analyzer.check_data_balance()
    analyzer.plot_data_balance(
        output_dir=args.output_dir,
        output_name=f"data_balance_{split}.png",
    )

    print("\n2. CHECKING ANNOTATION QUALITY...")
    issues = analyzer.check_annotation_quality()

    if args.visualize_augmentations or args.visualize_jpeg_artifacts:
        visualizer = AugmentationVisualizer(img_dir)
        sample_files = analyzer.df.sample(
            min(args.sample_size, len(analyzer.df)),
            random_state=args.seed,
        )["filename"].tolist()

        if args.visualize_augmentations:
            print(f"\n3. VISUALIZING AUGMENTATIONS ({len(sample_files)} samples)...")
            img_paths = []
            for i, filename in enumerate(sample_files, 1):
                img_path = Path(img_dir) / filename
                if img_path.exists():
                    print(f"   [{i}/{len(sample_files)}] {filename}")
                    visualizer.visualize_augmentations(str(img_path), output_dir=args.output_dir)
                    img_paths.append(str(img_path))

            if img_paths:
                print("   Creating batch visualization...")
                visualizer.visualize_batch(img_paths, output_dir=args.output_dir)

        if args.visualize_jpeg_artifacts:
            print(f"\n4. VISUALIZING JPEG ZOOM CHECKS ({len(sample_files)} samples)...")
            for i, filename in enumerate(sample_files, 1):
                img_path = Path(img_dir) / filename
                if img_path.exists():
                    print(f"   [{i}/{len(sample_files)}] {filename}")
                    visualizer.visualize_jpeg_zoom(str(img_path), output_dir=args.output_dir)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Split: {split}")
    print(f"Total samples: {len(analyzer.df)}")
    print(f"Missing files: {len(issues['missing_files'])}")
    print(f"Unreadable images: {len(issues['unreadable_images'])}")
    print(f"Invalid classes: {len(issues['invalid_classes'])}")
    print(f"Duplicates: {len(issues['duplicates'])}")
    print(f"Filename/label mismatches: {len(issues['filename_label_mismatch'])}")
    print(f"Very small images: {len(issues['small_images'])}")
    print(f"Output saved to: {args.output_dir}")
    print("=" * 70 + "\n")

    if any(issues.values()):
        print("RECOMMENDATIONS:")
        if issues["missing_files"]:
            print(f"   - Restore or remove {len(issues['missing_files'])} missing files")
        if issues["unreadable_images"]:
            print(f"   - Re-export {len(issues['unreadable_images'])} unreadable images")
        if issues["invalid_classes"]:
            print(f"   - Check {len(issues['invalid_classes'])} invalid labels")
        if issues["duplicates"]:
            print(f"   - Review {len(issues['duplicates'])} duplicate rows")
        if issues["filename_label_mismatch"]:
            print(f"   - Manually inspect {len(issues['filename_label_mismatch'])} label mismatches")
        if issues["small_images"]:
            print(f"   - Verify {len(issues['small_images'])} very small images")
    else:
        print("DATA LOOKS CLEAN!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze dataset balance, labels, augmentations and JPEG artifacts"
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="train",
        help="Which split to analyze",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--visualize_augmentations",
        action="store_true",
        help="Create augmentation visualizations",
    )
    parser.add_argument(
        "--visualize_jpeg_artifacts",
        action="store_true",
        help="Create zoomed crops to inspect JPEG artifacts",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=6,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for visual samples",
    )
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        analyze_split(split, args)


if __name__ == "__main__":
    main()

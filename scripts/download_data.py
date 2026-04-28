import os
import zipfile
import subprocess
from pathlib import Path


def download_from_kaggle(output_dir: str = "data/raw"):
    """
    Downloads NEU Surface Defect Dataset from Kaggle.
    Requirements:
      - pip install kaggle
      - kaggle.json in ~/.kaggle/  (from kaggle.com -> Account -> API)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("[Download] Downloading NEU dataset from Kaggle...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "kaustubhdikshit/neu-surface-defect-database",
        "-p", output_dir,
    ], check=True)

    zip_path = Path(output_dir) / "neu-surface-defect-database.zip"
    print(f"[Download] Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)

    zip_path.unlink()
    print(f"[Download] Done! Data saved to: {output_dir}")
    _print_structure(output_dir)


def _print_structure(base_dir: str):
    print("\n[Structure]")
    for root, dirs, files in os.walk(base_dir):
        level = root.replace(base_dir, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < 2:
            for f in files[:3]:
                print(f"{indent}  {f}")


if __name__ == "__main__":
    download_from_kaggle()
# Advanced ML Improvements

## Overview

This document summarizes the latest project update focused on data inspection and advanced machine learning techniques for improving training quality and model reliability.

The previous robustness work on ensemble inference, robust TTA, uncertainty estimation, and production endpoints is documented separately in [ROBUSTNESS_IMPROVEMENTS.md](ROBUSTNESS_IMPROVEMENTS.md).

---

## Implemented Updates

### 0. Clean Test Set vs Robustness Probes

**Status:** Implemented

The evaluation protocol now explicitly separates the official clean benchmark
from robustness experiments.

**Files:**
- `README.md`
- `scripts/robustness_eval.py`
- `src/evaluation/robustness.py`
- `src/datasets/transforms.py`

**Rules:**
- Training augmentations are used only on `train`.
- `val` and `test` use clean resize + normalization only.
- Noise, blur, brightness and JPEG compression are applied in memory for
  robustness probes and saved under separate report names.

**How to run:**
```powershell
python evaluate.py --checkpoint checkpoints/best_resnet50.pt
python scripts/robustness_eval.py --mode single --model resnet50 --checkpoint checkpoints/best_resnet50.pt --split test
```

**Generated outputs:**
- `reports/robustness_test_results.csv`
- `reports/robustness_test_plot.png`

---

### 1. Visual Exploration of Data Augmentations

**Status:** Implemented

Added a workflow to visually inspect data augmentations and verify that the generated transformations remain realistic for industrial defect images.

**Files:**
- `scripts/analyze_data.py`
- `src/datasets/data_analyzer.py`

**How to run:**
```powershell
python scripts/analyze_data.py --split train --visualize_augmentations --visualize_jpeg_artifacts
```

**Generated outputs:**
- `reports/augmentations_batch.png`
- `reports/augmentations_*.png`
- `reports/jpeg_zoom_*.png`
- `reports/data_balance_<split>.png`

**Purpose:**
- Check whether augmentations preserve defect visibility.
- Detect transformations that may distort industrial images too strongly.
- Inspect JPEG/compression artifacts with zoomed crops.
- Support better decisions before training.

---

### 1b. K-Fold Cross-Validation Workflow

**Status:** Implemented

Added a stratified K-fold workflow to measure performance variability across
multiple held-out folds instead of relying on a single train/val/test split.

**Files:**
- `scripts/prepare_kfold_splits.py`
- `scripts/run_kfold.py`
- `train.py`
- `evaluate.py`
- `src/training/trainer.py`
- `src/evaluation/metrics.py`

**How to run:**
```powershell
python scripts/prepare_kfold_splits.py --n_splits 5
python scripts/run_kfold.py --model resnet50
```

**Generated outputs:**
- `data/splits/kfold/fold_*/train.csv`
- `data/splits/kfold/fold_*/val.csv`
- `data/splits/kfold/fold_*/test.csv`
- `reports/kfold_summary_resnet50.csv`

**Purpose:**
- Check whether the reported accuracy is stable across folds.
- Report mean and standard deviation, not only one best split.
- Better expose overfitting when data is limited.

---

### 1c. Overfitting Diagnostics

**Status:** Implemented

The trainer now exports train/validation curves and a compact overfitting
summary after each run.

**Files:**
- `src/training/trainer.py`
- `src/evaluation/metrics.py`
- `evaluate.py`

**Generated outputs:**
- `reports/training_history_<run_name>.csv`
- `reports/training_summary_<run_name>.json`
- `reports/training_curves_<run_name>.png`
- `reports/<report_name>_report.json` with accuracy and classification report

**Purpose:**
- Compare train accuracy vs validation accuracy.
- Detect suspicious 100% scores with a measurable train/val gap.
- Keep confusion matrices and classification reports tied to each evaluation.

---

### 2. Data Balance Verification

**Status:** Implemented

Added dataset balance analysis to check whether defect classes are evenly represented across the dataset splits.

**Generated output:**
- `reports/data_balance.png`

**Purpose:**
- Identify class imbalance.
- Reduce the risk of biased model behavior.
- Help decide whether sampling or class weighting is needed.

The training pipeline also uses a weighted sampler and class weights to reduce the impact of imbalance during training.

---

### 3. Annotation and Data Quality Checks

**Status:** Implemented

Added dataset checks to detect potential data quality problems before model training.

**Checks included:**
- Missing image files.
- Invalid paths.
- Class distribution issues.
- Unreadable images.
- Filename/label mismatches.
- Very small images.
- Samples that should be manually reviewed.

**Purpose:**
- Reduce training noise.
- Improve confidence in the dataset.
- Highlight possible annotation problems, which are common in visual inspection tasks.

---

### 4. EMA Training Utility

**Status:** Integrated as optional training experiment

Added an Exponential Moving Average utility for stabilizing model weights during
training experiments, and wired it into the trainer.

**File:**
- `src/training/ema.py`
- `src/training/trainer.py`
- `train.py`
- `configs/config.yaml`

**Main class:**
- `EMAScheduler`

**Capabilities:**
- Maintains shadow EMA weights.
- Updates EMA weights after optimizer steps.
- Temporarily applies EMA weights during validation.
- Saves the best checkpoint using EMA weights when enabled.
- Can be enabled from config or CLI.

**How to run:**
```powershell
python train.py --model resnet50 --run_name resnet50_ema --use_ema --ema_decay 0.999
```

**Purpose:**
- Stabilize weight updates.
- Smooth training behavior.
- Improve generalization in future experiments.

---

### 5. LoRA Fine-Tuning Support

**Status:** Implemented

Added LoRA support for parameter-efficient fine-tuning.

**Files:**
- `src/training/lora.py`
- `configs/config.yaml`
- `train.py`
- `scripts/export_onnx.py`
- `scripts/generate_predictions.py`
- `scripts/gradcam.py`
- `api/inference.py`
- `api/main.py`

**Configuration:**
```yaml
training:
  use_lora: true
  lora_rank: 4
  lora_alpha: 32
  lora_dropout: 0.1
  lora_target_modules:
    - fc
    - classifier
```

**Purpose:**
- Fine-tune fewer parameters.
- Reduce memory usage.
- Keep training and inference compatible with the existing pipeline.

---

### 6. RF-DETER Training Utility

**Status:** Module implemented

Added RF-DETER utilities for training-time perturbation experiments.

**File:**
- `src/training/rf_deter.py`

**Main classes:**
- `RFDeterMixin`
- `RFDeterWrapper`

**Capabilities:**
- Adds controlled perturbations during training.
- Keeps inference clean by disabling perturbations outside training mode.
- Supports configurable perturbation strength.

**Purpose:**
- Explore robustness-oriented training.
- Reduce sensitivity to small input variations.
- Prepare the project for future training experiments under more realistic conditions.

---

## Summary

This update adds:
- Visual augmentation inspection.
- Dataset balance analysis.
- Annotation and data quality checks.
- EMA utility.
- LoRA fine-tuning support.
- RF-DETER training utility.

**Last Updated:** April 29, 2026

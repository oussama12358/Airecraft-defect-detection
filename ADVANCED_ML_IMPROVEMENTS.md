# Advanced ML Improvements

## Overview

This document summarizes the latest project update focused on data inspection and advanced machine learning techniques for improving training quality and model reliability.

The previous robustness work on ensemble inference, robust TTA, uncertainty estimation, and production endpoints is documented separately in [ROBUSTNESS_IMPROVEMENTS.md](ROBUSTNESS_IMPROVEMENTS.md).

---

## Implemented Updates

### 1. Visual Exploration of Data Augmentations

**Status:** Implemented

Added a workflow to visually inspect data augmentations and verify that the generated transformations remain realistic for industrial defect images.

**Files:**
- `scripts/analyze_data.py`
- `src/datasets/data_analyzer.py`

**How to run:**
```powershell
python scripts/analyze_data.py --split train --visualize_augmentations
```

**Generated outputs:**
- `reports/augmentations_batch.png`
- `reports/augmentations_*.png`

**Purpose:**
- Check whether augmentations preserve defect visibility.
- Detect transformations that may distort industrial images too strongly.
- Support better decisions before training.

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
- Samples that should be manually reviewed.

**Purpose:**
- Reduce training noise.
- Improve confidence in the dataset.
- Highlight possible annotation problems, which are common in visual inspection tasks.

---

### 4. EMA Training Utility

**Status:** Module implemented

Added an Exponential Moving Average utility for stabilizing model weights during training experiments.

**File:**
- `src/training/ema.py`

**Main class:**
- `EMAScheduler`

**Capabilities:**
- Maintains shadow EMA weights.
- Updates EMA weights after optimizer steps.
- Temporarily applies EMA weights during evaluation.
- Supports checkpoint save/load through `state_dict()`.

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

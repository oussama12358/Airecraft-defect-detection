# ✈️ Aircraft Defect Detection using Deep Learning

> A deep learning system for automatic detection of surface defects in aircraft and industrial materials using image classification models trained on the **NEU-DET** dataset.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset & Defect Classes](#-dataset--defect-classes)
- [Models Used](#-models-used)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Output](#-output)
- [Key Features](#-key-features)
- [Use Cases](#-use-cases)
- [Author](#-author)
- [License](#-license)

---

## 📌 Project Overview

This project focuses on **detecting and classifying surface defects** in steel and industrial materials using multiple deep learning architectures. The system supports end-to-end inference: from raw image input to annotated predictions with confidence scores.

The pipeline is designed for real-world deployment in aerospace and industrial quality control contexts.

---

## 🗂️ Dataset & Defect Classes

**Dataset:** [NEU-DET](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) — a standard benchmark for surface defect detection.

The model classifies **6 types of surface defects**:

| # | Class | Description |
|---|-------|-------------|
| 1 | `crazing` | Network of fine cracks on the surface |
| 2 | `inclusion` | Foreign material embedded in the surface |
| 3 | `patches` | Irregular patches on the material |
| 4 | `pitted_surface` | Small pits or cavities on the surface |
| 5 | `rolled-in_scale` | Scale pressed into the surface during rolling |
| 6 | `scratches` | Linear marks or grooves on the surface |

---

## 🧠 Models Used

| Model | Architecture | Accuracy |
|-------|-------------|----------|
| 🥉 Baseline CNN | Custom lightweight CNN | ~95.8% |
| 🥈 EfficientNet-B3 | Transfer learning (EfficientNet) | ~99.5% |
| 🥇 **ResNet50** | **Transfer learning (ResNet50)** | **100%** |

> ✅ **Best model: ResNet50** — achieves perfect classification on the NEU-DET test set.

---

## 📊 Results

- **Best model:** ResNet50 — 100% accuracy
- **Evaluation metrics:** Confusion matrix + per-class classification accuracy
- **Generalization:** Strong performance across all 6 defect categories on the NEU-DET benchmark

---

## 📁 Project Structure

```
aircraft-defect-detection/
│
├── checkpoints/                  # Saved model weights
│   ├── best_resnet50.pt
│   ├── best_efficientnet_b3.pt
│   └── best_baseline_cnn.pt
│
├── scripts/
│   └── generate_predictions.py   # Main inference script
│
├── assets/
│   └── results/                  # Output predictions
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/aircraft-defect-detection.git
cd aircraft-defect-detection
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 🥇 ResNet50 (Best — Recommended)
```bash
python scripts/generate_predictions.py --model checkpoints/best_resnet50.pt
```

### 🥈 EfficientNet-B3
```bash
python scripts/generate_predictions.py --model checkpoints/best_efficientnet_b3.pt
```

### 🥉 Baseline CNN
```bash
python scripts/generate_predictions.py --model checkpoints/best_baseline_cnn.pt
```

> 💡 The inference pipeline **automatically detects** the model architecture from the checkpoint name and loads the correct weights.

---

## 📤 Output

All predictions are saved to:

```
assets/results/
```

Each output image is annotated with:
- ✅ **Predicted defect class**
- 📊 **Confidence score**

---

## 🧠 Key Features

- **Multi-model support** — Seamlessly switch between CNN, ResNet50, and EfficientNet-B3
- **Automatic architecture detection** — Loads correct model based on checkpoint filename
- **Flexible dataset handling** — Supports both image folder and CSV-based dataset input
- **Robust checkpoint loading** — Handles `state_dict` loading with error recovery
- **Production-ready pipeline** — Designed for real industrial inspection workflows

---

## ✈️ Use Cases

This system is aligned with real-world applications in:

- 🛩️ **Aerospace inspection** — Automated surface quality checks for aircraft components
- 🏭 **Industrial quality control** — In-line defect detection on production lines
- 🔬 **Surface defect detection** — Research and benchmarking on NEU-DET and similar datasets

---

## 👨‍💻 Author

**Oussama Sghir**

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

# ✈️ Aircraft Defect Detection using Deep Learning

A deep learning system for automatic detection of surface defects in aircraft and industrial materials using image classification models trained on the NEU-DET dataset.

---

## 📌 Project Overview

This project focuses on detecting and classifying surface defects such as:

- crazing  
- inclusion  
- patches  
- pitted_surface  
- rolled-in_scale  
- scratches  

using multiple deep learning architectures.

The system supports:

- CNN baseline model  
- ResNet50  
- EfficientNet-B3  
- Multi-checkpoint inference pipeline  
- Flexible dataset loading (folder or CSV)

---

## 🧠 Models Used

| Model | Description | Performance |
|------|-------------|-------------|
| Baseline CNN | Simple CNN architecture | ~95.8% |
| ResNet50 | Transfer learning model | 100% accuracy |
| EfficientNet-B3 | Optimized deep model | ~99.5% |

---

## 📊 Results

- Best model: **ResNet50**
- Evaluation: Confusion matrix + classification accuracy
- Strong generalization on NEU-DET dataset

---

## 🧪 Inference Pipeline

The system automatically:

- Loads the correct model architecture
- Loads checkpoint weights (state_dict)
- Processes test images
- Outputs annotated predictions

Supports:

- Image folder input
- CSV-based dataset input

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run predictions
🥇 ResNet50
python scripts/generate_predictions.py --model checkpoints/best_resnet50.pt
🥈 EfficientNet-B3
python scripts/generate_predictions.py --model checkpoints/best_efficientnet_b3.pt
🥉 Baseline CNN
python scripts/generate_predictions.py --model checkpoints/best_baseline_cnn.pt
📁 Output

All predictions are saved in:

assets/results/

Each image contains:

predicted class
confidence score
🧠 Key Features
Multi-model support
Robust checkpoint loading
Flexible dataset handling
Production-style inference pipeline
Ready for industrial inspection use cases
✈️ Use Case

This project is aligned with real-world applications in:

Aerospace inspection
Industrial quality control
Surface defect detection systems

👨‍💻 Author
Oussama Sghir

License
MIT

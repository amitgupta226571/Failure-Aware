# Failure-Aware
# 📄 **README.md - Complete GitHub Documentation**

---

# 🧠 Failure-Aware Domain Generalization for Brain Tumor MRI Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **AI that knows when it's wrong: A safety-first approach to brain tumor detection across hospitals**

---

## 📌 **Overview**

This repository implements a **failure-aware domain generalization pipeline** for brain tumor MRI classification. Instead of just predicting tumors, the model also tells you when it might be wrong—because in healthcare, knowing your limits is as important as being right.

### 🔬 **What Makes This Different**

- **Domain Adaptation**: Works across different hospitals/scanners using DANN and MixStyle
- **Failure Prediction**: Learns to predict its own mistakes (no manual labeling needed)
- **Safety Metrics**: Introduces **False Safe Rate (FSR)** —a metric that measures dangerous overconfidence
- **6 Architectures**: Compare ResNet50, MobileNetV2, ConvNeXt, DenseNet121, Swin, DeiT
- **Clinical Ready**: Risk-coverage analysis and deployment guidelines included

---

## 📊 **Key Results**

| Model | Accuracy | Failure F1 | False Safe Rate | Best For |
|-------|----------|------------|-----------------|----------|
| **Swin-Tiny** | 90.4% | **0.506** | **31.8%** | 🏆 **Safety** |
| MobileNetV2 | **91.2%** | 0.219 | 70.7% | Accuracy |
| DenseNet121 | 82.0% | 0.474 | 50.0% | Balance |
| ConvNeXt | 78.8% | 0.142 | 87.8% | ❌ Not safe |
| ResNet50 | 65.4% | 0.426 | 39.2% | Low accuracy |

### 🎯 **The Big Takeaway**
> **Swin Transformer is statistically as accurate as MobileNetV2 (p=0.653) but catches 68% of errors vs 29%.** 
> *Higher accuracy doesn't always mean safer.*

---

## 🏗️ **Pipeline Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│  BR35H (Source) ──► Augmentation ──► Train (80%)          │
│                    └─► Validate (20%)                      │
│  BTD (Target) ────► Split ──► DANN Target (50%)           │
│                              └─► Final Test (50%)          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     MODEL ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│  Input MRI (224×224)                                       │
│       ↓                                                    │
│  Backbone (ResNet/MobileNet/ConvNeXt/DenseNet/Swin/DeiT)  │
│       ↓                                                    │
│  ┌───────┴───────┐                                         │
│  ↓               ↓                                         │
│ Classification  Failure Head                               │
│     Head        (LayerNorm + LeakyReLU + Dropout)          │
│  ┌───────┐     ┌─────────────────┐                         │
│  │Tumor? │     │ Safety Score   │                         │
│  └───────┘     └─────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   DOMAIN GENERALIZATION                     │
├─────────────────────────────────────────────────────────────┤
│  • DANN: Gradient reversal to learn domain-invariant features│
│  • MixStyle: Random style mixing for data augmentation     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 **Getting Started**

### **Prerequisites**
```bash
Python 3.8+
CUDA 11.0+ (for GPU training)
```

### **Installation**
```bash
git clone https://github.com/yourusername/failure-aware-brain-mri.git
cd failure-aware-brain-mri
pip install -r requirements.txt
```

### **Dataset Setup**
```bash
# Download datasets
# BR35H: [link]
# BTD: [link]

# Organize as:
data/
├── BR35H/
│   └── train/
│       ├── yes/     # Tumor images
│       └── no/      # No tumor images
└── BTD/
    └── test/
        ├── glioma/      # Tumor
        ├── meningioma/  # Tumor
        ├── pituitary/   # Tumor
        └── no_tumor/    # No tumor
```

### **Run Training**
```bash
# Train all models
python train.py

# Train specific model
python train.py --model resnet50 --epochs 30

# Evaluation only
python evaluate.py --checkpoint checkpoints/resnet50_best.pth
```

---

## 📊 **Results & Visualizations**

### **Failure Head Performance**
The failure head learns to separate correct vs incorrect predictions:

```
Before training:
Correct: 0.50  Incorrect: 0.52  (barely any difference)

After training:
Correct: 0.15  Incorrect: 0.65  (clear separation!)
```

### **Grad-CAM Visualizations**
*Heatmaps showing where the model looks*

| Original | Grad-CAM Overlay |
|----------|------------------|
| ![MRI](assets/mri_sample.png) | ![GradCAM](assets/gradcam.png) |

### **Risk-Coverage Curve**
*Shows how many cases you need to review to achieve target safety*

![Risk-Coverage](assets/risk_coverage.png)

---

## 📁 **Repository Structure**

```
failure-aware-brain-mri/
├── data/                       # Dataset loaders and preprocessing
│   ├── dataset.py              # Custom dataset classes
│   └── transforms.py           # Augmentations
│
├── models/                     # Model architectures
│   ├── backbones.py            # Feature extractors
│   ├── failure_head.py         # Failure prediction head
│   └── dann.py                 # Domain adaptation layers
│
├── training/                   # Training utilities
│   ├── train.py                # Main training loop
│   ├── loss.py                 # Loss functions
│   └── scheduler.py            # Learning rate schedules
│
├── evaluation/                 # Evaluation scripts
│   ├── metrics.py              # Classification metrics
│   ├── failure_metrics.py      # Failure prediction metrics
│   └── visualize.py            # Visualizations
│
├── configs/                    # Configuration files
│   └── config.yaml
│
├── checkpoints/                # Saved models
├── plots/                      # Generated visualizations
├── results/                    # Evaluation results
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## 🛠️ **Key Technical Details**

### **Loss Function**
```
L_total = L_class + 3.0 × L_failure + 0.2 × L_domain

L_failure = BCE(target, pred) + diversity_reg + margin_loss
target = (1 - confidence) + 0.5 × (is_incorrect)
```

### **Architecture Highlights**
- **Failure Head**: LayerNorm (not BatchNorm) for consistent train/eval behavior
- **DANN**: Progressive alpha schedule (0 → 1)
- **Separate Learning Rates**: Failure head gets 10× higher LR for faster adaptation

### **Safety Metrics Explained**
| Metric | What It Measures | Target |
|--------|------------------|--------|
| **False Safe Rate (FSR)** | % of errors dangerously trusted | <25% |
| **Failure Recall** | % of errors caught | >70% |
| **Failure F1** | Balanced safety performance | >0.6 |

---

## 📝 **Results Summary**

| Model | Accuracy | Failure F1 | FSR | Sensitivity | Specificity |
|-------|----------|------------|-----|--------------|-------------|
| Swin-Tiny | 90.4% | **0.506** | **31.8%** | 87.9% | 96.1% |
| MobileNetV2 | **91.2%** | 0.219 | 70.7% | 89.0% | 96.1% |
| DenseNet121 | 82.0% | 0.474 | 50.0% | 75.9% | 95.6% |
| ConvNeXt | 78.8% | 0.142 | 87.8% | 71.1% | 96.1% |
| ResNet50 | 65.4% | 0.426 | 39.2% | 52.8% | 93.6% |

---

## 🏥 **Clinical Deployment Guidelines**

| Scenario | Recommended Model | Why |
|----------|-------------------|-----|
| **Maximum Safety** | Swin-Tiny | Lowest FSR (32%), highest failure recall (68%) |
| **Maximum Accuracy** | MobileNetV2 | 91% accuracy, but requires safety monitoring |
| **Balanced** | DenseNet121 | Good balance of accuracy and safety |

### **FSR Thresholds for Deployment**
- **FSR < 25%**: ✅ Safe for autonomous deployment
- **FSR 25-50%**: ⚠️ Human oversight required
- **FSR 50-75%**: ❌ Not recommended
- **FSR > 75%**: 🔴 Dangerous—do not deploy

---

## 📚 **Citation**

If you use this work, please cite:

```bibtex
@article{yourname2025failure,
  title={Failure-Aware Domain Generalization for Brain Tumor MRI Classification},
  author={Your Name},
  journal={},
  year={2025}
}
```

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- BR35H and BTD datasets
- PyTorch and HuggingFace Timm teams
- Domain adaptation research community

---

## 📧 **Contact**

Questions? Suggestions? Reach out:
- **Email**: amitgupta226571@gmail.com
- **GitHub**: [@AmitGupta](https://github.com/amitgupta226571)

---

**Built with ❤️ for safer medical AI**

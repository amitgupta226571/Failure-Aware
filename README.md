# 📄 README.md - Complete GitHub Documentation

Here is the properly formatted README file for your repository.


# 🧠 Failure-Aware Domain Generalization for Brain Tumor MRI Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **AI that knows when it's wrong: A safety-first approach to brain tumor detection across hospitals**

---

## 📌 Overview

This repository implements a **failure-aware domain generalization pipeline** for brain tumor MRI classification. Instead of just predicting tumors, the model also tells you when it might be wrong—because in healthcare, knowing your limits is as important as being right.

### 🔬 What Makes This Different

- **Domain Adaptation**: Works across different hospitals/scanners using DANN and MixStyle
- **Failure Prediction**: Learns to predict its own mistakes (no manual labeling needed)
- **Safety Metrics**: Introduces **False Safe Rate (FSR)** —a metric that measures dangerous overconfidence
- **6 Architectures**: Compare ResNet50, DenseNet121, EfficientNet-B0, ConvNeXt, Swin, DeiT
- **Two Evaluation Protocols**: Standard Calibration (safety priority) and SAFE Evaluation (sensitivity priority)
- **Clinical Ready**: Risk-coverage analysis and deployment guidelines included

---

## 📊 Key Results

### Standard Calibration (Safety Priority)

| Model | Accuracy | Sensitivity | Specificity | FSR ↓ | Failure F1 | Failure Recall | AUC |
|-------|----------|-------------|-------------|-------|------------|----------------|-----|
| **Swin-Tiny** | **90.09%** | 87.20% | 96.55% | **7.69%** | 0.400 | **92.31%** | **0.978** |
| EfficientNet-B0 | 85.82% | 80.79% | 97.04% | 11.83% | 0.414 | 88.17% | 0.973 |
| DeiT | 85.67% | 81.46% | 95.07% | 0.00%* | 0.282 | 100.00%* | 0.969 |
| ResNet50 | 76.37% | 68.21% | 94.58% | 13.55% | 0.528 | 86.45% | 0.941 |
| DenseNet121 | 72.56% | 61.81% | 96.55% | 30.56% | 0.598 | 69.44% | 0.950 |
| ConvNeXt | 70.88% | 59.60% | 96.06% | 66.49% | 0.357 | 33.51% | 0.891 |

*Note: DeiT had only 3 incorrect predictions, making FSR statistically unstable.

### SAFE Evaluation (Sensitivity Priority)

| Model | Accuracy | Sensitivity | Specificity | FSR | Failure Recall |
|-------|----------|-------------|-------------|-----|----------------|
| **Swin-Tiny** | **93.75%** | **93.16%** | 95.07% | 14.63% | 85.37% |
| DeiT | 91.16% | 89.40% | 95.07% | 25.86% | 74.14% |
| EfficientNet-B0 | 88.11% | 84.55% | 96.06% | 34.62% | 65.38% |

### 🎯 The Big Takeaway

> **Swin Transformer achieves the lowest False Safe Rate (7.69%)**, meaning fewer than 8 out of every 100 errors would be dangerously trusted. This meets the proposed clinical safety threshold of <15% FSR.

> **Accuracy alone is misleading** — ConvNeXt (70.88% accuracy) has FSR of 66.49%, while ResNet50 (76.37% accuracy) has FSR of 13.55%. A less accurate model can be clinically safer.

---

## 🏗️ Pipeline Architecture

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
│  Backbone (6 architectures supported)                      │
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

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
CUDA 11.0+ (for GPU training)
```

### Installation

```bash
git clone https://github.com/yourusername/failure-aware-brain-mri.git
cd failure-aware-brain-mri
pip install -r requirements.txt
```

### Dataset Setup

```bash
# Download datasets
# BR35H: https://data.mendeley.com/datasets/8zwbr82bbk/1
# BTD: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-dataset

# Organize as:
data/
├── BR35H/
│   └── train/
│       ├── yes/     # Tumor images (1200)
│       └── no/      # No tumor images (1200)
└── BTD/
    └── test/
        ├── glioma/      # Tumor
        ├── meningioma/  # Tumor
        ├── pituitary/   # Tumor
        └── notumor/     # No tumor
```

### Run Training

```bash
# Train all models
python train.py

# Train specific model
python train.py --model swin_tiny --epochs 30

# Evaluation only
python evaluate.py --checkpoint checkpoints/swin_tiny_best.pth
```

---

## 📊 Results & Visualizations

### Failure Head Performance

The failure head learns to separate correct vs incorrect predictions:

| Model | Correct Mean | Incorrect Mean | Separation | T-test p-value | Correlation | Status |
|-------|--------------|----------------|------------|----------------|-------------|--------|
| EfficientNet-B0 | 0.1712 | 0.5107 | 0.3395 | <0.001 | 0.472 | Healthy |
| Swin-Tiny | 0.0953 | 0.0664 | -0.0289 | 0.7805 | 0.350 | Weak* |
| DeiT-Small | 0.1033 | 0.5121 | 0.4088 | 0.0016 | 0.330 | Healthy |
| DenseNet121 | 0.1708 | 0.4788 | 0.3080 | <0.001 | 0.323 | Healthy |
| ResNet50 | 0.1724 | 0.3868 | 0.2144 | <0.001 | 0.318 | Healthy |
| ConvNeXt-Tiny | 0.1832 | 0.5955 | 0.4123 | 0.0065 | 0.088 | Weak |

*Swin-T shows weak statistical differentiation (p=0.78) but maintains good correlation (0.35) and excellent AUC (0.978). The weak result is likely due to small error sample (n=4).

### Risk-Coverage Analysis

For Swin-T: Rejecting just 10% of uncertain cases reduces FSR from 7.69% to below 3%.

---

## 📁 Repository Structure

```
failure-aware-brain-mri/
├── data/                       # Dataset loaders and preprocessing
│   ├── dataset.py              # Custom dataset classes
│   └── transforms.py           # Augmentations
│
├── models/                     # Model architectures
│   ├── backbones.py            # Feature extractors (6 architectures)
│   ├── failure_head.py         # Failure prediction head with LayerNorm
│   └── dann.py                 # Domain adaptation layers
│
├── training/                   # Training utilities
│   ├── train.py                # Main training loop
│   ├── loss.py                 # Multi-task loss function
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

## 🛠️ Key Technical Details

### Loss Function

```
L_total = L_class + 3.0 × L_failure + 0.2 × L_domain

L_failure = BCE(target, pred) + 0.5 × diversity_reg + 5.0 × margin_loss
target = (1 - confidence) + 0.5 × (is_incorrect)
```

### Architecture Highlights

- **Failure Head**: LayerNorm (not BatchNorm) for consistent train/eval behavior under domain shift
- **DANN**: Progressive alpha schedule (0 → 1) with gradient reversal
- **Separate Learning Rates**: Failure head gets 10× higher LR for faster adaptation
- **Margin Loss**: Forces incorrect failure scores ≥ correct failure scores + 0.15

### Safety Metrics Explained

| Metric | What It Measures | Target | Swin-T Result |
|--------|------------------|--------|---------------|
| **False Safe Rate (FSR)** | % of errors dangerously trusted | <15% | **7.69%** ✅ |
| **Failure Recall** | % of errors caught | >80% | **92.31%** ✅ |
| **Failure F1** | Balanced safety performance | >0.4 | 0.400 |

### Two Evaluation Protocols

| Protocol | Method | Best For | Swin-T FSR |
|----------|--------|----------|------------|
| **Standard Calibration** | Optimized class threshold (0.867) | Confirmatory diagnosis (safety priority) | 7.69% |
| **SAFE Evaluation** | Temperature scaling (0.121) + percentile threshold | Screening / emergency (sensitivity priority) | 14.63% |

---

## 📝 Results Summary

### Statistical Significance (McNemar's Test)

| Comparison | p-value | Significant? |
|------------|---------|--------------|
| Swin-T vs EfficientNet-B0 | 0.003 | ✅ Yes |
| Swin-T vs ResNet50 | <0.001 | ✅ Yes |
| Swin-T vs DenseNet121 | <0.001 | ✅ Yes |
| Transformers vs CNNs | 0.012 | ✅ Yes |

---

## 🏥 Clinical Deployment Guidelines

| Scenario | Recommended Protocol | Recommended Model | Key Metric |
|----------|---------------------|-------------------|------------|
| **Confirmatory Diagnosis** | Standard Calibration | Swin-Tiny | FSR = 7.69% |
| **Screening / Emergency** | SAFE Evaluation | Swin-Tiny | Sensitivity = 93.16% |
| **Resource-constrained** | Standard Calibration | EfficientNet-B0 | FSR = 11.83% |

### Clinical Workflow

```
MRI Scan → Model Prediction + Failure Score
                              ↓
                    if failure_score < 0.057:
                        → AUTOMATIC ACCEPT
                    else:
                        → FLAG FOR RADIOLOGIST REVIEW
```

### FSR Thresholds for Deployment

- **FSR < 10%**: ✅ Safe for autonomous deployment (Swin-T: 7.69%)
- **FSR 10-15%**: ⚠️ Acceptable with oversight
- **FSR 15-25%**: ⚠️ Human oversight required
- **FSR > 25%**: ❌ Not recommended for clinical use

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@article{gupta2025failure,
  title={Failure-Aware Domain Generalization for Brain Tumor MRI Classification},
  author={Gupta, Amit and Kumar, Vinod},
  journal={},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- BR35H and BTD datasets
- PyTorch and HuggingFace Timm teams
- Domain adaptation research community

---

## 📧 Contact

Questions? Suggestions? Reach out:
- **Email**: amitgupta226571@gmail.com
- **GitHub**: [@amitgupta226571](https://github.com/amitgupta226571)

---

**Built with ❤️ for safer medical AI**
```

---

## Quick Copy Instructions

1. Click the copy button on the code block above
2. Create a new file named `README.md` in your repository root
3. Paste the content
4. Save and commit

# 🧠 Failure-Aware Domain Generalization for Brain Tumor MRI Classification

AI that knows when it's wrong: A safety-first approach to brain tumor detection across hospitals.

---

## What This Project Does

This repository implements a failure-aware domain generalization pipeline for brain tumor MRI classification. The model doesn't just predict tumors—it also tells you when it might be wrong. Because in healthcare, knowing your limits is as important as being right.

The approach works across different hospitals and scanners using DANN and MixStyle for domain adaptation. The failure head learns to predict mistakes without requiring manual labeling. We introduce False Safe Rate (FSR) as a metric that measures dangerous overconfidence. The code supports six architectures: ResNet50, DenseNet121, EfficientNet-B0, ConvNeXt, Swin Transformer, and DeiT.

---

## Key Results

### Standard Calibration (Safety Priority)

| Model | Accuracy | Sensitivity | Specificity | FSR | Failure F1 | Failure Recall | AUC |
|-------|----------|-------------|-------------|-----|------------|----------------|-----|
| Swin-Tiny | 90.09% | 87.20% | 96.55% | 7.69% | 0.400 | 92.31% | 0.978 |
| EfficientNet-B0 | 85.82% | 80.79% | 97.04% | 11.83% | 0.414 | 88.17% | 0.973 |
| DeiT | 85.67% | 81.46% | 95.07% | 0.00%* | 0.282 | 100.00%* | 0.969 |
| ResNet50 | 76.37% | 68.21% | 94.58% | 13.55% | 0.528 | 86.45% | 0.941 |
| DenseNet121 | 72.56% | 61.81% | 96.55% | 30.56% | 0.598 | 69.44% | 0.950 |
| ConvNeXt | 70.88% | 59.60% | 96.06% | 66.49% | 0.357 | 33.51% | 0.891 |

*DeiT had only 3 incorrect predictions, making FSR statistically unstable.

### SAFE Evaluation (Sensitivity Priority)

| Model | Accuracy | Sensitivity | Specificity | FSR | Failure Recall |
|-------|----------|-------------|-------------|-----|----------------|
| Swin-Tiny | 93.75% | 93.16% | 95.07% | 14.63% | 85.37% |
| DeiT | 91.16% | 89.40% | 95.07% | 25.86% | 74.14% |
| EfficientNet-B0 | 88.11% | 84.55% | 96.06% | 34.62% | 65.38% |

### The Big Takeaway

The Swin Transformer achieves the lowest False Safe Rate at 7.69%, meaning fewer than 8 out of every 100 errors would be dangerously trusted. This meets the proposed clinical safety threshold of less than 15% FSR.

Accuracy alone can be misleading. ConvNeXt achieves 70.88% accuracy but has an FSR of 66.49%, while ResNet50 achieves 76.37% accuracy with an FSR of only 13.55%. A less accurate model can actually be clinically safer.

---

## Pipeline Architecture

The pipeline follows a clear flow. The BR35H dataset serves as the source domain and is split into 80% training and 20% validation with augmentation applied to the training set. The BTD dataset is the target domain and is split equally into DANN training and final test sets.

The model architecture takes an input MRI of 224×224 pixels and passes it through a backbone feature extractor. From there, it branches into two heads. The classification head predicts whether a tumor is present. The failure head, which uses LayerNorm, LeakyReLU, and dropout, outputs a safety score indicating how likely the prediction is to be wrong.

For domain generalization, we use two techniques. DANN applies gradient reversal to learn domain-invariant features. MixStyle performs random style mixing for data augmentation.

---

## Getting Started

### Prerequisites

You need Python 3.8 or higher and CUDA 11.0 or higher for GPU training.

### Installation

Clone the repository and install dependencies:

```
git clone https://github.com/yourusername/failure-aware-brain-mri.git
cd failure-aware-brain-mri
pip install -r requirements.txt
```

### Dataset Setup

Download the BR35H dataset from Mendeley Data and the BTD dataset from Kaggle. Organize them as follows:

```
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

To train all models, run `python train.py`. To train a specific model, use `python train.py --model swin_tiny --epochs 30`. For evaluation only, use `python evaluate.py --checkpoint checkpoints/swin_tiny_best.pth`.

---

## Results and Visualizations

### Failure Head Performance

The failure head learns to separate correct from incorrect predictions. EfficientNet-B0 shows the best performance with a separation of 0.3395, a p-value less than 0.001, and a correlation of 0.472. Swin-Tiny shows weak statistical differentiation with a p-value of 0.7805 but maintains good correlation at 0.350 and excellent AUC at 0.978. The weak result is likely due to the small error sample of only 4 incorrect predictions.

### Risk-Coverage Analysis

For the Swin Transformer, rejecting just 10% of uncertain cases reduces the False Safe Rate from 7.69% to below 3%.

---

## Repository Structure

The repository is organized into several directories. The data directory contains dataset loaders and preprocessing code. The models directory holds the backbone implementations, the failure head with LayerNorm, and the DANN layers. The training directory includes the main training loop, loss functions, and learning rate schedules. The evaluation directory contains metrics calculation and visualization scripts. Configurations are stored in the configs directory. Checkpoints, plots, and results are saved in their respective directories.

---

## Key Technical Details

### Loss Function

The total loss combines three components: classification loss, failure loss, and domain loss. The failure loss itself includes binary cross-entropy, diversity regularization, and margin loss. The target for the failure head is calculated as one minus confidence plus 0.5 times the incorrect indicator.

### Architecture Highlights

The failure head uses LayerNorm instead of BatchNorm to maintain consistent behavior under domain shift. DANN uses a progressive alpha schedule from 0 to 1 with gradient reversal. The failure head gets 10 times higher learning rate for faster adaptation. The margin loss forces incorrect failure scores to be at least 0.15 higher than correct failure scores.

### Safety Metrics

False Safe Rate measures the percentage of errors dangerously trusted, with a target below 15%. Failure Recall measures the percentage of errors caught, with a target above 80%. Failure F1 provides a balanced measure of safety performance.

### Two Evaluation Protocols

Standard calibration uses an optimized class threshold of 0.867 and is best for confirmatory diagnosis where safety is the priority. This gives an FSR of 7.69% for Swin-T. SAFE evaluation uses temperature scaling of 0.121 and a percentile threshold, best for screening and emergency situations where sensitivity is the priority. This gives an FSR of 14.63% for Swin-T.

---

## Clinical Deployment Guidelines

For confirmatory diagnosis, use standard calibration with the Swin-Tiny model, which gives an FSR of 7.69%. For screening and emergency situations, use SAFE evaluation with Swin-Tiny, which gives a sensitivity of 93.16%. For resource-constrained settings, use standard calibration with EfficientNet-B0, which gives an FSR of 11.83%.

### Clinical Workflow

The model processes an MRI scan and produces both a prediction and a failure score. If the failure score is below 0.057, the prediction is automatically accepted. If the failure score is 0.057 or higher, the case is flagged for radiologist review.

### FSR Thresholds for Deployment

An FSR below 10% is safe for autonomous deployment, and Swin-T achieves 7.69%. An FSR between 10% and 15% is acceptable with oversight. An FSR between 15% and 25% requires human oversight. An FSR above 25% is not recommended for clinical use.

---

## Citation

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

## License

This project is licensed under the MIT License.

---

## Acknowledgments

We thank the creators of the BR35H and BTD datasets, the PyTorch and HuggingFace Timm teams, and the domain adaptation research community.

---

## Contact

For questions or suggestions, reach out to amitgupta226571@gmail.com or visit https://github.com/amitgupta226571.

---

Built with ❤️ for safer medical AI.

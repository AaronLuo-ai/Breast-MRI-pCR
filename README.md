cat << 'EOF' > README.md
# Breast MRI pCR Prediction

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?style=flat&logo=pytorchlightning&logoColor=white)](https://www.pytorchlightning.ai/)

This repository contains a deep learning pipeline designed to predict **Pathologic Complete Response (pCR)** using features extracted from multi-phase Dynamic Contrast-Enhanced (DCE) MRI.

## 📌 Project Overview
- **Data**: 3-Phase MRI (Pre-contrast, Post-1, Post-2) processed as temporal stacks.
- **Model**: Residual Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout.
- **Validation**: Stratified 5-Fold Cross-Validation ($n=104$).
- **Metrics**: Primary: **AUC** | Secondary: **MCC** and **Accuracy**.

## 🛠️ Architecture Details
The model handles high-dimensional features (3,456 dims) using a residual architecture to ensure robust gradient flow:
- **Residual Blocks**: Two layers with skip connections and LeakyReLU activation.
- **Regularization**: 
  - **Gaussian Noise**: $\sigma=0.01$ added to features during training to improve generalization.
  - **Dropout**: Dual-stage (0.5 in residual blocks, 0.3 in classification head).
  - **Weight Decay**: L2 regularization ($1e-2$) via Adam optimizer.
- **Loss Function**: `BCEWithLogitsLoss`.



## 📂 Folder Structure
This project is designed to be placed within the `pillar-pretrain` directory.

```text
pillar-pretrain/
└── Breast-MRI-pCR/
    ├── data/               # Dataloaders:
    │   ├── RGB-Concatenation
    │   ├── RGB-Stacking
    │   └── Single-phase (Pre/Po1/Po2)
    ├── model/              # mlp.py (PcrPredictor LightningModule)
    ├── batch_script/       # SLURM templates (sbatch_template.sh)
    ├── reports/            # Generated Excel metrics per fold
    ├── checkpoint/         # Saved .ckpt files (Best Val AUC)
    └── run.py              # Main training/validation entry point

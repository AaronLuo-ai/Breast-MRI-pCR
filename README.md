cat << 'EOF' > README.md
# Breast MRI pCR Prediction

This repository contains a deep learning pipeline designed to predict **Pathologic Complete Response (pCR)** using features extracted from multi-phase DCE-MRI.

## 📌 Project Overview
- **Data**: 3-Phase MRI (Pre-contrast, Post-1, Post-2) processed as temporal stacks.
- **Model**: A Residual Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout.
- **Validation**: Stratified 5-Fold Cross-Validation to ensure robustness on a small medical dataset ($n=104$).
- **Metrics**: Primary metric is **AUC (Area Under the ROC Curve)**, alongside MCC and Accuracy.

## 🛠️ Architecture Details
- **Residual Blocks**: Two residual layers to facilitate gradient flow and prevent vanishing gradients.
- **Regularization**: 
  - **Gaussian Noise**: $\sigma=0.01$ added to features during training.
  - **Dropout**: Configurable rates for residual blocks and the classification head.
  - **Weight Decay**: L2 regularization ($1e-2$) applied via Adam optimizer.
- **Loss Function**: Binary Cross Entropy with Logits (BCEWithLogitsLoss).

## 📂 Folder Structure
- `data/`: Dataloaders for MRI feature extraction and phase stacking.
- `model/`: `mlp.py` containing the `PcrPredictor` LightningModule.
- `reports/`: Automatically generated Excel sheets for each fold capturing best-epoch metrics.
- `checkpoint/`: Saved `.ckpt` files for the best performing model in each fold.

## 🚀 How to Run
1. **Environment Setup**:
   ```bash
   conda create -n onco -c conda-forge python=3.12
   conda activate onco
   pip install -r requirements.txt

import torch
import torch.nn as nn
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
import re
import wandb
import pytorch_lightning as L
import os
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
# Assuming these are in your local files
from data.dataloader import MriFeatureDataset, MriFeatureDataset2
from model.mlp import PcrPredictor

def get_dist(labels):
    counts = np.bincount(labels)
    perc = (counts / len(labels)) * 100
    return f"Non-pCR: {perc[0]:.1f}% | pCR: {perc[1]:.1f}% (n={len(labels)})"


def run_mlp_cross_validation(base_path, excel_path, config_path, ckpt_path, 
                             batch_size, lr, max_epochs, weight_decay,
                             residual_dropout, head_dropout, hidden_dim, stack_rgb):
    full_dataset = None
    dir_name = "rgb_stack" if stack_rgb else "rgb_concat"
    if stack_rgb:
        # 1. Initialize Dataset (RGB stacking of Pre, Po1, Po2)
        full_dataset = MriFeatureDataset(
            feature_path=base_path,
            response_path=excel_path,
            model_config_path=config_path,
            checkpoint_path=ckpt_path
        )
    else: 
        # RGB concatenation of Pre, Po1, Po2
        full_dataset = MriFeatureDataset2(
            feature_path=base_path,
            response_path=excel_path,
            model_config_path=config_path,
            checkpoint_path=ckpt_path
        )

    cv_auc_scores = []

    # 2. Extract Raw Features (Concatenating Pre, Po1, Po2)
    all_features = []
    all_labels = []
    all_fold_reports = []

    for i in range(len(full_dataset)):
        feats, label = full_dataset[i] 
        raw_feat = feats.flatten() 
        all_features.append(raw_feat.numpy())
        all_labels.append(label.item())

    X = np.array(all_features)
    Y = np.array(all_labels)
    
    input_dim = X.shape[1]

    # 3. Setup Stratified 5-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y)):
        # --- CLASS DISTRIBUTION CHECK ---
        y_train, y_val = Y[train_idx], Y[val_idx]

        print(f"\n--- Fold {fold+1}/5 ---")
        print(f"  Train Dist: {get_dist(y_train)}")
        print(f"  Val Dist:   {get_dist(y_val)}")

        # 4. Standardize (Crucial for MLPs when not using PCA)
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train_raw)
        # X_val_scaled = scaler.transform(X_val_raw)
        X_train_scaled = X_train_raw
        X_val_scaled = X_val_raw
        # 5. Sampler for Class Imbalance
        # class_counts = np.bincount(y_train)
        # class_weights = 1. / class_counts
        # sample_weights = [class_weights[l] for l in y_train]
        # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        # 6. DataLoaders (Using Scaled Raw Features)
        train_ds = TensorDataset(torch.from_numpy(X_train_scaled).float(), torch.from_numpy(y_train).long())
        val_ds = TensorDataset(torch.from_numpy(X_val_scaled).float(), torch.from_numpy(y_val).long())

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=None, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # 7. Logger & Model (input_dim is now 3456)
        wandb_logger = WandbLogger(
            project="Breast-MRI-pCR", 
            name=f"fold_{fold+1}_lr_{lr}_bs_{batch_size}_rd_{residual_dropout}_hd_{head_dropout}_{dir_name}",
            group=f"LR_{lr}_BS_{batch_size}", # Grouping helps in the WandB UI
            job_type="grid_search"
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_auc",
            dirpath=f"/scratch/aaron.l/checkpoint/{dir_name}_mlp_fold_{fold+1}_lr_{lr}_wd_{weight_decay}_rd_{residual_dropout}_hd_{head_dropout}",
            # {epoch} and {val_auc} are automatically filled by Lightning
            filename="best-performance-{epoch:02d}-{val_auc:.3f}",
            mode="max",
            save_top_k=1,
        )

        model = PcrPredictor(input_dim=input_dim, lr=lr, weight_decay=weight_decay, residual_dropout=residual_dropout, head_dropout=head_dropout, hidden_dim=hidden_dim) 
        
        # 8. Trainer
        trainer = L.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            deterministic=True,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
            num_sanity_val_steps=0,
            check_val_every_n_epoch=1
        )

        trainer.fit(model, train_loader, val_loader)

        print("Evaluating best model on validation set...")

        trainer.validate(model, val_loader, ckpt_path="best", verbose=False)
        
        # 3. Extract metrics from the callback_metrics dictionary
        metrics = trainer.callback_metrics
        best_model_path = checkpoint_callback.best_model_path
        epoch_match = re.search(r"epoch=(\d+)", best_model_path)
        best_epoch = int(epoch_match.group(1)) if epoch_match else -1

        # 4. Construct the detailed report for this fold
        fold_report = {
            "Fold": fold + 1,
            "Best Epoch": best_epoch,
            # Merge all hyperparameters from the model
            **model.hparams, 
            "Batch Size": batch_size,
            "Max Epochs": max_epochs,
            "Validation AUC": checkpoint_callback.best_model_score.item(),
            "Validation Loss": metrics.get("val_loss").item() if "val_loss" in metrics else "N/A",
            "Training AUC": metrics.get("train_auc").item() if "train_auc" in metrics else "N/A",
            "Training Loss": metrics.get("train_loss").item() if "train_loss" in metrics else "N/A",
            "Checkpoint Path": best_model_path
        }

        # 5. Save to Excel

        best_auc = checkpoint_callback.best_model_score.item()
        cv_auc_scores.append(best_auc)
        all_fold_reports.append(fold_report)

        print(f"Fold {fold+1} Best AUC: {best_auc:.4f}")
        print(f"Best model for Fold {fold+1} saved at: {best_model_path}")
        wandb.finish()
    
    avg_auc = np.mean(cv_auc_scores)
    std_auc = np.std(cv_auc_scores)

    base_output_path = "/scratch/aaron.l/output"
    folder_name = f"{dir_name}_avg_auc_{avg_auc:.3f}_bs_{batch_size}_lr_{lr}_wd_{weight_decay}_rd_{residual_dropout}_hd_{head_dropout}"
    final_report_dir = os.path.join(base_output_path, folder_name)
    
    # Create the directory
    os.makedirs(final_report_dir, exist_ok=True)
    # Save each fold's report into this new directory
    for i, report in enumerate(all_fold_reports):
        file_name = f"fold_{report['Fold']}_auc_{report['Validation AUC']:.3f}.xlsx"
        save_path = os.path.join(final_report_dir, file_name)
        pd.DataFrame([report]).to_excel(save_path, index=False)
        
    print("\n" + "="*30)
    print(f"FINAL CV RESULTS (5-Fold)")
    print(f"Individual Scores: {[round(s, 3) for s in cv_auc_scores]}")
    print(f"Mean AUC: {avg_auc:.3f}")
    print(f"Std Dev:  {std_auc:.3f}")
    print(f"Reportable Result: {avg_auc:.2f} ± {std_auc:.2f}")
    print(f"All reports saved in: {final_report_dir}")
    print("="*30)


if __name__ == "__main__":
    L.seed_everything(42, workers=True)
    run_mlp_cross_validation(
        base_path="/home/aaron.l/FeatureExtraction/MriExtraction/mri_features",
        excel_path="/home/aaron.l/VisualizeDir/MultimodalPilotDataset_v1_DEID.xlsx",
        config_path="/home/aaron.l/Pillar/model.config.yaml",
        ckpt_path="/home/aaron.l/Pillar/pillar-pretrain/model.safetensors",
        batch_size=4,
        lr=3e-5,
        max_epochs=200,
        weight_decay=1e-2,
        residual_dropout=0.1,
        head_dropout=0.1,
        hidden_dim=256,
        stack_rgb=False
    )

    # lr_arr = [3e-5, 1e-4, 3e-4, 1e-3]
    # batch_size_arr = [4, 8, 16]
    # residual_dropout_arr = [0.5, 0.1, 0.2, 0.3, 0.0]
    # head_dropout_arr = [0.5, 0.1, 0.2, 0.3, 0.0] # Renamed from dead_dropout

    # for lr in lr_arr:
    #     for batch_size in batch_size_arr:
    #         for residual_dropout in residual_dropout_arr:
    #             for head_dropout in head_dropout_arr:
    #                 for stack_rgb in [True, False]:
    #                     run_mlp_cross_validation(
    #                         base_path="/home/aaron.l/FeatureExtraction/MriExtraction/mri_features",
    #                         excel_path="/home/aaron.l/VisualizeDir/MultimodalPilotDataset_v1_DEID.xlsx",
    #                         config_path="/home/aaron.l/Pillar/model.config.yaml",
    #                         ckpt_path="/home/aaron.l/Pillar/pillar-pretrain/model.safetensors",
    #                         batch_size=batch_size,
    #                         lr=lr,
    #                         max_epochs=200,
    #                         weight_decay=1e-2,
    #                         residual_dropout=residual_dropout,
    #                         head_dropout=head_dropout,
    #                         hidden_dim=256,
    #                         stack_rgb=stack_rgb
    #                     )

# TODO: Use RGB stacking PRE PO1 PO2 as 3 channels instead of flattening. 
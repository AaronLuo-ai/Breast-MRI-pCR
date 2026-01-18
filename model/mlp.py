import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import torchmetrics
from torchvision.ops import sigmoid_focal_loss # Import the official ops

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_prob=0.4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        # The 'Residual' part: Add the original input back to the output
        return self.activation(x + self.block(x))

class PcrPredictor(L.LightningModule):
    def __init__(self, input_dim=384, hidden_dim=256, lr=1e-4, residual_dropout=0.5, head_dropout=0.3, weight_decay=1e-2):
        super().__init__()
        self.save_hyperparameters()

        # Architecture
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.res_block1 = ResidualBlock(hidden_dim, dropout_prob=residual_dropout)
        self.res_block2 = ResidualBlock(hidden_dim, dropout_prob=residual_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.train_auc = torchmetrics.AUROC(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")
        self.val_mcc = torchmetrics.MatthewsCorrCoef(num_classes=2)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Data Augmentation (Noise)
        # x = x + torch.randn_like(x) * 0.01
        
        # Get logits (don't apply sigmoid yet!)
        logits = self(x).squeeze(-1) 
        
        # 1. Use torchvision Focal Loss
        # alpha=0.25 (Balances the classes)
        # gamma=2.0 (Down-weights easy examples)
        # loss = sigmoid_focal_loss(
        #     inputs=logits, 
        #     targets=y.float(), 
        #     alpha=0.25, 
        #     gamma=2.0, 
        #     reduction="mean"
        # )

        loss = F.binary_cross_entropy_with_logits(logits, y.float())

        preds = torch.sigmoid(logits)
        self.train_acc(preds, y)
        self.train_auc(preds, y)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x).squeeze(-1)
        
        # 2. Use Focal Loss for validation as well
        # loss = sigmoid_focal_loss(
        #     inputs=logits, 
        #     targets=y.float(), 
        #     alpha=0.25, 
        #     gamma=2.0, 
        #     reduction="mean"
        # )

        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        preds = torch.sigmoid(logits)
        self.val_auc(preds, y)
        self.val_mcc(preds, y)
        
        self.log_dict({
            "val_loss": loss,
            "val_auc": self.val_auc,
            "val_mcc": self.val_mcc
        }, on_epoch=True, on_step=False, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay # Fix this line
        )
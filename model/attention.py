import torch
import torch.nn as nn
import pytorch_lightning as L
import torchmetrics

class AttentionPcrPredictor(L.LightningModule):
    def __init__(self, phase_dim=1152, hidden_dim=256, nhead=8, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # 1. Projection: Bring each phase into a common embedding space
        self.phase_projection = nn.Sequential(
            nn.Linear(phase_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )

        # 2. Attention Layer: Learns relationships between Pre, Po1, and Po2
        # batch_first=True means input is (Batch, Seq, Feature)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=nhead, 
            batch_first=True,
            dropout=0.3
        )

        # 3. Classifier: Takes the "attended" features and makes a decision
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, 1)
        )

        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")

    def forward(self, x):
        # x input is flat [Batch, 3456]
        # Reshape to [Batch, 3, 1152] -> (Pre, Po1, Po2)
        batch_size = x.shape[0]
        x = x.view(batch_size, 3, self.hparams.phase_dim)

        # Project phases to hidden_dim
        x = self.phase_projection(x) # [Batch, 3, 256]

        # Multi-Head Attention
        # attn_output shape: [Batch, 3, 256]
        attn_output, _ = self.attention(x, x, x)

        # Global Average Pooling across the 3 phases
        # Instead of flattening, we average the insights from the 3 timepoints
        out = torch.mean(attn_output, dim=1) # [Batch, 256]

        return self.classifier(out)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        preds = torch.sigmoid(logits)
        self.val_auc(preds, y)
        self.log_dict({"val_loss": loss, "val_auc": self.val_auc}, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-2)
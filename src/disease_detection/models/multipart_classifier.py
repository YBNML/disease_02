"""4-부위 (leaf/branch/fruit/stem) multi-head 3-class 분류기.

각 부위마다 3-class head (`absent`, `normal`, `defect`) — 독립 CrossEntropyLoss 로
합산하는 multi-task 구조. Backbone 은 ResNet18 공유.

출력 Tensor 형태: `(B, NUM_PARTS, NUM_STATES) = (B, 4, 3)`.
"""
from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import ResNet18_Weights, resnet18

NUM_PARTS: int = 4    # leaf, branch, fruit, stem
NUM_STATES: int = 3   # absent, normal, defect


class MultiPartDefectClassifier(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        t_max: int = 30,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()  # 공유 feature
        self.backbone = backbone
        self.head = nn.Linear(in_features, NUM_PARTS * NUM_STATES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)                      # (B, F)
        logits = self.head(feat)                     # (B, NUM_PARTS * NUM_STATES)
        return logits.view(-1, NUM_PARTS, NUM_STATES)

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # logits (B, 4, 3), y (B, 4)
        # 부위별 CrossEntropy를 평균.
        loss = 0.0
        for i in range(NUM_PARTS):
            loss = loss + F.cross_entropy(logits[:, i, :], y[:, i])
        return loss / NUM_PARTS

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        preds = logits.argmax(dim=-1)  # (B, 4)
        acc_per_part = (preds == y).float().mean(dim=0)  # (4,)
        macro_acc = acc_per_part.mean()
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc_macro", macro_acc, on_epoch=True, prog_bar=True)
        for i, part_name in enumerate(("leaf", "branch", "fruit", "stem")):
            self.log(f"val/acc_{part_name}", acc_per_part[i], on_epoch=True)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.t_max)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

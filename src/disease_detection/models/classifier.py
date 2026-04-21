"""ResNet18 기반 이진 분류기 LightningModule.

화상병용·범용 결함용으로 두 인스턴스를 독립 학습. 구조는 동일, 학습 데이터·체크포인트만 분리.
"""
from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import ResNet18_Weights, resnet18


class PlantDefectClassifier(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        pos_weight: float | None = None,
        t_max: int = 30,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 1)
        self.model = backbone

        pw = (
            torch.tensor([pos_weight], dtype=torch.float32)
            if pos_weight is not None
            else None
        )
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        preds = (torch.sigmoid(logits) >= 0.5).long()
        acc = (preds == y).float().mean()
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.t_max)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

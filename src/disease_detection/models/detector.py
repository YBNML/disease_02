"""torchvision Faster R-CNN ResNet50-FPN v2 LightningModule."""
from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNNModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 4,  # background + leaf + stem + fruit
        lr: float = 0.005,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        lr_step: int = 10,
        lr_gamma: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model = model
        # prediction cache 축적용 (on_validation_epoch_end에서 비움).
        self._val_cache: list[tuple[list[dict], list[dict]]] = []

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())
        self.log_dict(
            {f"train/{k}": v for k, v in loss_dict.items()},
            on_step=True, on_epoch=True, prog_bar=False,
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Lightning이 진입 시 self.eval()을 호출하므로 self.model은 이미 eval 모드.
        # 수동 토글은 train 모드로 되돌려 후속 val 배치의 contract를 깨뜨리므로 금지.
        images, targets = batch
        with torch.no_grad():
            preds = self.model(images)
        # 실제 mAP 계산은 evaluate 스크립트에서. 여기선 샘플 수만 로깅하여 sanity check.
        # GPU 메모리 누적 방지를 위해 CPU로 detach하여 저장.
        self._val_cache.append(
            (
                [{k: v.detach().cpu() for k, v in p.items()} for p in preds],
                [{k: v.detach().cpu() for k, v in t.items()} for t in targets],
            )
        )

    def on_validation_epoch_end(self) -> None:
        count = len(self._val_cache)
        self.log("val/batches", float(count), prog_bar=False)
        self._val_cache = []

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = SGD(
            params,
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = StepLR(
            optimizer,
            step_size=self.hparams.lr_step,
            gamma=self.hparams.lr_gamma,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

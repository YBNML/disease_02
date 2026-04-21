"""torchvision Faster R-CNN ResNet50-FPN v2 LightningModule."""
from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchmetrics.detection import MeanAveragePrecision
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
        # val mAP를 스트리밍 집계. `validation_step`마다 update,
        # `on_validation_epoch_end`에서 compute → log → reset.
        self._val_map = MeanAveragePrecision(iou_type="bbox", class_metrics=False)

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
        images, targets = batch
        with torch.no_grad():
            preds = self.model(images)
        # CPU로 detach한 사본으로 MAP 업데이트 — GPU 누적 방지.
        preds_cpu = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
        targets_cpu = [{k: v.detach().cpu() for k, v in t.items()} for t in targets]
        self._val_map.update(preds_cpu, targets_cpu)

    def on_validation_epoch_end(self) -> None:
        raw = self._val_map.compute()
        # 체크포인트·조기 종료가 참조하는 주요 지표. mode=max.
        self.log("val/map_50", raw["map_50"], prog_bar=True)
        self.log("val/map", raw["map"], prog_bar=False)
        self.log("val/map_75", raw["map_75"], prog_bar=False)
        self._val_map.reset()

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

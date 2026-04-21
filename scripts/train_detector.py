#!/usr/bin/env python3
"""Faster R-CNN 학습 Hydra 진입점."""
from __future__ import annotations

import json
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from disease_detection.data.aihub import load_aihub_split
from disease_detection.data.detection_dataset import DetectionDataset
from disease_detection.data.transforms import (
    build_detector_eval_transform,
    build_detector_train_transform,
)
from disease_detection.models.detector import FasterRCNNModule
from disease_detection.utils.seeding import set_seed


def _detection_collate(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


def _build_loader(root: Path, indices: list[int], transform, batch_size: int, num_workers: int, shuffle: bool):
    entries = []
    for crop_name in ("pear", "apple"):
        entries.extend(load_aihub_split(root / crop_name))
    subset = [entries[i] for i in indices]
    ds = DetectionDataset(subset, transform=transform)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_detection_collate,
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    print(OmegaConf.to_yaml(cfg))

    aihub_root = Path(cfg.paths.dataset_root) / "raw" / "aihub"
    splits_path = Path(cfg.paths.dataset_root) / "splits" / "detector_split.json"
    split = json.loads(splits_path.read_text(encoding="utf-8"))

    train_loader = _build_loader(
        aihub_root,
        split["train"],
        build_detector_train_transform(),
        cfg.data.batch_size,
        cfg.data.num_workers,
        shuffle=True,
    )
    val_loader = _build_loader(
        aihub_root,
        split["val"],
        build_detector_eval_transform(),
        cfg.data.batch_size,
        cfg.data.num_workers,
        shuffle=False,
    )

    module = FasterRCNNModule(
        num_classes=cfg.model.num_classes,
        lr=cfg.model.lr,
        momentum=cfg.model.momentum,
        weight_decay=cfg.model.weight_decay,
        lr_step=cfg.model.lr_step,
        lr_gamma=cfg.model.lr_gamma,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{cfg.paths.models_root}/detector/{cfg.run.name}",
            monitor=cfg.trainer.checkpoint.monitor,
            mode=cfg.trainer.checkpoint.mode,
            save_top_k=cfg.trainer.checkpoint.save_top_k,
            filename=cfg.trainer.checkpoint.filename,
        )
    ]
    if cfg.trainer.get("early_stopping"):
        callbacks.append(
            EarlyStopping(
                monitor=cfg.trainer.early_stopping.monitor,
                patience=cfg.trainer.early_stopping.patience,
                mode=cfg.trainer.early_stopping.mode,
            )
        )

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", None),
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        limit_train_batches=cfg.trainer.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.trainer.get("limit_val_batches", 1.0),
        callbacks=callbacks,
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()

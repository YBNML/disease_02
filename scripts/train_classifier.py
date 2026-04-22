#!/usr/bin/env python3
"""PlantDefectClassifier 학습 Hydra 진입점. experiment preset에서 kind 분기."""
from __future__ import annotations

import json
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from disease_detection.data.classification_dataset import (
    ClassificationCropDataset,
    build_defect_crops,
    build_fireblight_crops,
)
from disease_detection.data.transforms import (
    build_classifier_eval_transform,
    build_classifier_train_transform,
)
from disease_detection.models.classifier import PlantDefectClassifier
from disease_detection.utils.seeding import set_seed


def _build_items(cfg: DictConfig):
    aihub_root = Path(cfg.paths.dataset_root) / "raw" / "aihub"
    kind = cfg.classifier.kind
    if kind == "fireblight":
        items = []
        for crop_name in ("pear", "apple"):
            items.extend(build_fireblight_crops(aihub_root / crop_name))
        split_name = "classifier_fireblight_split.json"
    elif kind == "defect":
        vlm_path = Path(cfg.classifier.vlm_jsonl)
        items = []
        for crop_name in ("pear", "apple"):
            items.extend(
                build_defect_crops(
                    aihub_root / crop_name,
                    vlm_path,
                    defect_threshold=cfg.classifier.defect_threshold,
                )
            )
        split_name = "classifier_defect_split.json"
    else:
        raise ValueError(f"Unknown classifier.kind: {kind}")
    return items, split_name


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    print(OmegaConf.to_yaml(cfg))

    items, split_name = _build_items(cfg)
    splits_path = Path(cfg.paths.dataset_root) / "splits" / split_name
    split = json.loads(splits_path.read_text(encoding="utf-8"))

    train_items = [items[i] for i in split["train"]]
    val_items = [items[i] for i in split["val"]]

    train_ds = ClassificationCropDataset(
        train_items, transform=build_classifier_train_transform()
    )
    val_ds = ClassificationCropDataset(
        val_items, transform=build_classifier_eval_transform()
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    module = PlantDefectClassifier(
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        pos_weight=cfg.classifier.get("pos_weight"),
        t_max=cfg.model.t_max,
    )

    ckpt_dir = f"{cfg.paths.models_root}/classifier_{cfg.classifier.kind}/{cfg.run.name}"
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
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
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        limit_train_batches=cfg.trainer.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.trainer.get("limit_val_batches", 1.0),
        callbacks=callbacks,
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()

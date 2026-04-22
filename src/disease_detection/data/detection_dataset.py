"""Faster R-CNN 학습·추론용 Dataset (single-class plant ROI).

AIhub 데이터에는 plant part (leaf/stem/fruit) 라벨이 없으므로 detector는 단일 클래스
"plant_roi" 만을 학습한다. 배경 포함 `num_classes=2`.

torchvision detection 레퍼런스 규격을 따름:
    target = {
        "boxes":  tv_tensors.BoundingBoxes(xyxy, ...),
        "labels": int64 Tensor (전부 1),
        "image_id": int64 Tensor [idx],
        "area":   float Tensor (transform 이후 재계산),
        "iscrowd": int64 Tensor (zeros),
    }
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors

from .aihub import AIhubImage, load_aihub_split

PART_CATEGORIES: dict[str, int] = {"plant_roi": 1}
"""`AIhubBox.category` → Faster R-CNN label id. 배경=0, plant_roi=1."""

NUM_CLASSES: int = 1 + len(PART_CATEGORIES)
"""FasterRCNNModule(num_classes=...)에 넘겨야 하는 값. 배경 1 + 전경 N."""


class DetectionDataset(Dataset):
    def __init__(
        self,
        entries: list[AIhubImage],
        transform: Callable | None = None,
    ) -> None:
        self.entries = entries
        self.transform = transform

    @classmethod
    def from_aihub_root(
        cls, root: Path, transform: Callable | None = None
    ) -> "DetectionDataset":
        return cls(load_aihub_split(root), transform=transform)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        entry = self.entries[idx]
        with Image.open(entry.image_path) as src:
            pil = src.convert("RGB")
        img = tv_tensors.Image(pil)

        if entry.boxes:
            xyxy = torch.tensor([b.xyxy for b in entry.boxes], dtype=torch.float32)
            labels = torch.tensor(
                [PART_CATEGORIES[b.category] for b in entry.boxes], dtype=torch.int64
            )
        else:
            xyxy = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        boxes = tv_tensors.BoundingBoxes(
            xyxy, format="XYXY", canvas_size=(entry.height, entry.width)
        )
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        if self.transform is not None:
            img, target = self.transform(img, target)

        # `area`는 transform 이후 최종 bbox 기준으로 계산 — geometric aug로 크기 바뀜.
        b = target["boxes"]
        target["area"] = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        return img, target

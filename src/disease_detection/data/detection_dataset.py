"""Faster R-CNN 학습·추론용 Dataset.

torchvision detection reference 규격을 따른다: `target` = dict with `boxes`, `labels`,
`image_id`, `area`, `iscrowd`. v2 transforms는 `tv_tensors.BoundingBoxes` 를 처리.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors

from .aihub import AIhubImage, load_aihub_split

PART_CATEGORIES: dict[str, int] = {"leaf": 1, "stem": 2, "fruit": 3}


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
        pil = Image.open(entry.image_path).convert("RGB")
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

        # `area`는 transform 이후에 계산해야 geometric augmentation (resize, crop,
        # scale jitter 등) 적용된 bbox 기준이 된다. COCO eval의 small/medium/large
        # 분류에도 이 값이 쓰이므로 pre-transform 계산은 금지.
        b = target["boxes"]
        target["area"] = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        return img, target

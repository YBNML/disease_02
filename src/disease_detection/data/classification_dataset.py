"""Classifier용 bbox crop Dataset.

두 가지 분류 태스크를 지원:

1. **화상병 분류 (binary)** — `build_fireblight_items`:
   - AIhub 이미지 단위 `disease_code != 0` → 1
   - 출력: list[CropItem] (scalar label 0/1)

2. **범용 결함 분류 (4-부위 × 3-class multi-head)** — `build_defect_items`:
   - VLM v2 JSONL 라벨 기반 (`labels.<part>` ∈ {defect, normal, absent})
   - 출력: list[MultiPartCropItem] (4 부위 × 상태 벡터)

`ClassificationCropDataset` 은 scalar label 용, `MultiPartCropDataset` 은 vector 용.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors

from .aihub import load_aihub_split

# Multi-part 분류기의 부위 순서 & 상태 인코딩.
PLANT_PARTS: tuple[str, ...] = ("leaf", "branch", "fruit", "stem")
PART_STATES: dict[str, int] = {"absent": 0, "normal": 1, "defect": 2}
"""각 부위의 3-class: 0=absent (부위 없음), 1=normal, 2=defect.

multi-head 분류기는 부위마다 CrossEntropyLoss over 3-class 를 독립 계산한다.
"""


@dataclass(frozen=True)
class CropItem:
    """이진 분류용 crop 아이템 (화상병 분류기)."""

    image_path: Path
    xyxy: tuple[float, float, float, float]
    crop: str
    label: int  # 0 = 정상, 1 = 화상병


@dataclass(frozen=True)
class MultiPartCropItem:
    """4-부위 multi-head 분류용 crop 아이템 (범용 결함 분류기).

    `part_states[i]` ∈ {0,1,2} 이고 순서는 `PLANT_PARTS` 와 일치.
    """

    image_path: Path
    xyxy: tuple[float, float, float, float]
    crop: str
    part_states: tuple[int, int, int, int]


def build_fireblight_items(aihub_root: Path) -> list[CropItem]:
    """AIhub 이미지의 `disease_code != 0` 을 이진 라벨로 취해 bbox crop 아이템 생성.

    이미지당 1 bbox이므로 엔트리 수 == bbox 수.
    """
    entries = load_aihub_split(aihub_root)
    items: list[CropItem] = []
    for entry in entries:
        for box in entry.boxes:
            items.append(
                CropItem(
                    image_path=entry.image_path,
                    xyxy=box.xyxy,
                    crop=entry.crop,
                    label=int(entry.fireblight),
                )
            )
    return items


def build_defect_items(
    aihub_root: Path,
    vlm_labels_jsonl: Path,
) -> list[MultiPartCropItem]:
    """VLM v2 JSONL → `MultiPartCropItem`.

    JSONL 각 줄은 이미지 1장(bbox 1개)에 대한 4부위 라벨을 담는다:
        {
          "image_path": "...",
          "image_sha256": "...",
          "crop": "pear|apple",
          "parts": {
            "leaf":   {"state": "defect|normal|absent", ...},
            "branch": {...},
            "fruit":  {...},
            "stem":   {...}
          },
          ...
        }

    AIhub 엔트리 + JSONL 을 image stem으로 매칭. JSONL 에 없는 이미지는 스킵.
    """
    state_by_stem: dict[str, dict[str, int]] = {}
    with vlm_labels_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            stem = Path(obj["image_path"]).stem
            parts_raw = obj.get("parts", {}) or {}
            encoded: dict[str, int] = {}
            for part in PLANT_PARTS:
                info = parts_raw.get(part, {}) or {}
                state_str = str(info.get("state", "absent")).lower()
                if state_str not in PART_STATES:
                    raise ValueError(
                        f"Unknown part state '{state_str}' for {part} in {vlm_labels_jsonl}"
                    )
                encoded[part] = PART_STATES[state_str]
            state_by_stem[stem] = encoded

    entries = load_aihub_split(aihub_root)
    items: list[MultiPartCropItem] = []
    for entry in entries:
        stem = entry.image_path.stem
        if stem not in state_by_stem:
            continue
        enc = state_by_stem[stem]
        for box in entry.boxes:
            items.append(
                MultiPartCropItem(
                    image_path=entry.image_path,
                    xyxy=box.xyxy,
                    crop=entry.crop,
                    part_states=tuple(enc[p] for p in PLANT_PARTS),
                )
            )
    return items


class ClassificationCropDataset(Dataset):
    """이진 분류용 — `CropItem` → (Tensor, int)."""

    def __init__(
        self,
        items: list[CropItem],
        transform: Callable | None = None,
    ) -> None:
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple:
        item = self.items[idx]
        with Image.open(item.image_path) as src:
            pil = src.convert("RGB")
        x1, y1, x2, y2 = item.xyxy
        crop = pil.crop((x1, y1, x2, y2))
        img = tv_tensors.Image(crop)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(item.label)


class MultiPartCropDataset(Dataset):
    """4-부위 multi-head 용 — `MultiPartCropItem` → (Tensor, LongTensor[4])."""

    def __init__(
        self,
        items: list[MultiPartCropItem],
        transform: Callable | None = None,
    ) -> None:
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple:
        import torch

        item = self.items[idx]
        with Image.open(item.image_path) as src:
            pil = src.convert("RGB")
        x1, y1, x2, y2 = item.xyxy
        crop = pil.crop((x1, y1, x2, y2))
        img = tv_tensors.Image(crop)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(item.part_states, dtype=torch.long)

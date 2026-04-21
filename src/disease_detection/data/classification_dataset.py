"""ResNet18 학습·추론용 분류기 crop Dataset.

화상병 분류기와 범용 결함 분류기가 같은 Dataset 클래스를 공유. 라벨 생성 함수만 분리:
- `build_fireblight_crops` — AIhub 원본 이진 라벨 기반, 모든 bbox에 이미지 단위 라벨 적용.
- `build_defect_crops` — VLM severity JSONL을 이미지·부위별로 매칭, 임계값으로 이진화.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors

from .aihub import AIhubBox, AIhubImage, load_aihub_split


@dataclass(frozen=True)
class CropItem:
    """단일 분류기 학습 아이템."""

    image_path: Path
    xyxy: tuple[float, float, float, float]
    plant_part: str
    crop: str
    label: int  # 0 or 1


def build_fireblight_crops(aihub_root: Path) -> list[CropItem]:
    entries = load_aihub_split(aihub_root)
    items: list[CropItem] = []
    for entry in entries:
        for box in entry.boxes:
            items.append(
                CropItem(
                    image_path=entry.image_path,
                    xyxy=box.xyxy,
                    plant_part=box.category,
                    crop=entry.crop,
                    label=int(entry.fireblight),
                )
            )
    return items


def build_defect_crops(
    aihub_root: Path,
    vlm_labels_jsonl: Path,
    defect_threshold: int = 4,
) -> list[CropItem]:
    """VLM JSONL을 `(image_stem, plant_part) → severity` 맵으로 변환 후 매칭.

    이미지+부위 조합에 VLM 라벨이 있는 경우만 CropItem 생성. 없는 조합은 스킵.
    """
    sev_map: dict[tuple[str, str], int] = {}
    with vlm_labels_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            stem = Path(obj["image_path"]).stem
            part = obj["plant_part"]
            sev_map[(stem, part)] = int(obj["severity"])

    entries = load_aihub_split(aihub_root)
    items: list[CropItem] = []
    for entry in entries:
        for box in entry.boxes:
            key = (entry.image_path.stem, box.category)
            if key not in sev_map:
                continue
            sev = sev_map[key]
            label = 1 if sev >= defect_threshold else 0
            items.append(
                CropItem(
                    image_path=entry.image_path,
                    xyxy=box.xyxy,
                    plant_part=box.category,
                    crop=entry.crop,
                    label=label,
                )
            )
    return items


class ClassificationCropDataset(Dataset):
    """CropItem 리스트를 Tensor로 로딩."""

    def __init__(
        self,
        items: list[CropItem],
        transform: Callable | None = None,
    ) -> None:
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        pil = Image.open(item.image_path).convert("RGB")
        x1, y1, x2, y2 = item.xyxy
        crop = pil.crop((x1, y1, x2, y2))
        img = tv_tensors.Image(crop)
        if self.transform is not None:
            img = self.transform(img)
        else:
            # 기본: uint8 tensor 그대로 반환
            img = torch.as_tensor(list(crop.getdata()), dtype=torch.uint8).reshape(
                crop.height, crop.width, 3
            ).permute(2, 0, 1)
        return img, int(item.label)

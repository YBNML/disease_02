"""AIhub 과수원 화상병 데이터셋 파서.

실제 AIhub 배포 스키마가 확정되기 전까진 JSON 포맷을 가정. 스키마가 달라지면
`_load_annotation_json` 을 교체하거나 format-specific loader를 추가.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AIhubBox:
    """단일 부위 bbox + 카테고리."""

    category: str  # leaf / stem / fruit
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class AIhubImage:
    """한 장 이미지의 완전한 어노테이션."""

    image_path: Path
    crop: str  # "pear" or "apple"
    width: int
    height: int
    fireblight: int  # AIhub 원본 이진 라벨 (0 / 1)
    boxes: tuple[AIhubBox, ...] = field(default_factory=tuple)


def _load_annotation_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_entry(image_path: Path, ann: dict) -> AIhubImage:
    img_meta = ann["image"]
    boxes: list[AIhubBox] = []
    for a in ann.get("annotations", []):
        x, y, w, h = a["bbox"]
        boxes.append(
            AIhubBox(
                category=a["category"],
                xyxy=(float(x), float(y), float(x + w), float(y + h)),
            )
        )
    return AIhubImage(
        image_path=image_path,
        crop=ann.get("meta", {}).get("crop", "unknown"),
        width=int(img_meta["width"]),
        height=int(img_meta["height"]),
        fireblight=int(ann.get("labels", {}).get("fireblight", 0)),
        boxes=tuple(boxes),
    )


def load_aihub_split(root: Path) -> list[AIhubImage]:
    """`root/images/*.jpg` + `root/annotations/*.json` 쌍으로부터 항목 수집.

    어노테이션 파일 이름은 이미지 파일의 stem과 동일해야 한다.
    """
    images_dir = root / "images"
    ann_dir = root / "annotations"
    entries: list[AIhubImage] = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        ann_path = ann_dir / f"{img_path.stem}.json"
        if not ann_path.exists():
            continue
        ann = _load_annotation_json(ann_path)
        entries.append(_parse_entry(img_path, ann))
    return entries

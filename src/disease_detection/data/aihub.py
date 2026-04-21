"""AIhub 과수원 화상병 데이터셋 파서.

실제 AIhub 배포 스키마가 확정되기 전엔 JSON 포맷을 기본 가정. 다른 포맷(XML, CSV 등)
대응은 `load_aihub_split` 에 `annotation_loader` / glob 인자를 주입해서 확장한다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


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
    boxes: tuple[AIhubBox, ...] = ()


AnnotationLoader = Callable[[Path], dict]


def _load_annotation_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_entry(image_path: Path, ann: dict) -> AIhubImage:
    """Assumes COCO-style bbox `[x, y, w, h]`; converts to xyxy."""
    img_meta = ann["image"]
    boxes: list[AIhubBox] = []
    for a in ann.get("annotations", []):
        bbox = a["bbox"]
        if len(bbox) != 4:
            raise ValueError(f"bbox must have 4 elements (x,y,w,h): {bbox}")
        x, y, w, h = bbox
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


def load_aihub_split(
    root: Path,
    *,
    image_globs: tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png", "*.JPG"),
    annotation_suffix: str = ".json",
    annotation_loader: AnnotationLoader = _load_annotation_json,
    strict: bool = False,
) -> list[AIhubImage]:
    """`root/images/` + `root/annotations/` 쌍으로부터 항목 수집.

    각 이미지 파일의 stem과 동일한 이름의 어노테이션 파일을 매칭한다. 포맷·확장자
    차이는 인자로 조정한다 — 기본값은 JSON + 흔한 이미지 확장자 4종.

    Args:
        root: `images/`와 `annotations/` 하위 디렉토리를 가진 루트 경로.
        image_globs: 이미지로 인정할 glob 패턴 튜플.
        annotation_suffix: 어노테이션 파일 확장자 (선행 점 포함).
        annotation_loader: 어노테이션 파일 → dict 변환 함수. 기본은 JSON 파서.
        strict: True면 어노테이션 누락 시 FileNotFoundError, False면 조용히 스킵.

    Raises:
        ValueError: 어노테이션 구조 불일치 (파일 경로를 메시지에 포함).
        FileNotFoundError: strict=True인데 어노테이션 누락.
    """
    images_dir = root / "images"
    ann_dir = root / "annotations"
    entries: list[AIhubImage] = []
    image_paths: list[Path] = []
    for pattern in image_globs:
        image_paths.extend(images_dir.glob(pattern))
    for img_path in sorted(set(image_paths)):
        ann_path = ann_dir / f"{img_path.stem}{annotation_suffix}"
        if not ann_path.exists():
            if strict:
                raise FileNotFoundError(f"어노테이션 누락: {ann_path}")
            continue
        ann = annotation_loader(ann_path)
        try:
            entries.append(_parse_entry(img_path, ann))
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(
                f"어노테이션 파싱 실패 ({ann_path}): {exc}"
            ) from exc
    return entries

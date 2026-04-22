"""AIhub 과수화상병 촬영 이미지 데이터셋 파서.

실제 AIhub 라벨 스키마 (2025 기준 관측):

    {
      "description": {
        "image": "<basename>.jpg",
        "width":  <int>,
        "height": <int>,
        ...
      },
      "annotations": {
        "disease": <int>,     # 0 = 정상, 1~N = 질병 sub-type (화상병 등)
        "crop":    <int>,     # 1 = pear(배), 2 = apple(사과)
        "points":  [
          {"xtl": <int>, "ytl": <int>, "xbr": <int>, "ybr": <int>}
        ]
      }
    }

- 이미지당 bbox는 **정확히 1개** (관측된 모든 샘플 기준)
- bbox는 `xtl/ytl/xbr/ybr` (xyxy 형식, pixel 좌표)
- `disease` 는 image-level 라벨 — bbox 내부에 병변이 국소적으로 있는지 여부는 제공되지 않음
- plant part (leaf/stem/fruit) 라벨은 존재하지 않음

본 파서는 이 스키마를 `AIhubImage` 데이터클래스로 정규화한다.
Training / Validation 디렉토리는 상위 split 구분일 뿐, 우리는 자체 stratified split을
사용하므로 파서는 크롭·split 혼합 디렉토리에서 바로 로드 가능하게 설계한다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

CROP_CODE_TO_NAME: dict[int, str] = {1: "pear", 2: "apple"}
"""AIhub `annotations.crop` 정수 → 우리가 쓰는 작물 문자열."""


@dataclass(frozen=True)
class AIhubBox:
    """단일 plant ROI bbox.

    AIhub에 부위(leaf/stem/fruit) 카테고리가 없으므로 `category`는 모든 엔트리가
    `"plant_roi"`로 고정된다 (단일 클래스 detector용).
    """

    category: str
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class AIhubImage:
    """한 장 이미지의 정규화된 어노테이션."""

    image_path: Path
    crop: str                  # "pear" / "apple"
    width: int
    height: int
    disease_code: int          # AIhub 원본 `disease` 필드 (0=정상, >0=질병 sub-type)
    fireblight: int            # 이진화: `disease_code != 0` → 1, else 0
    boxes: tuple[AIhubBox, ...] = ()


AnnotationLoader = Callable[[Path], dict]


def _load_annotation_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_entry(image_path: Path, ann: dict) -> AIhubImage:
    """AIhub JSON dict → AIhubImage.

    Raises:
        ValueError: 스키마 필수 필드 누락·타입 불일치.
    """
    desc = ann["description"]
    labels = ann["annotations"]

    width = int(desc["width"])
    height = int(desc["height"])
    crop_code = int(labels.get("crop", 0))
    crop_name = CROP_CODE_TO_NAME.get(crop_code, "unknown")
    disease_code = int(labels.get("disease", 0))

    boxes: list[AIhubBox] = []
    for p in labels.get("points", []):
        xtl = float(p["xtl"])
        ytl = float(p["ytl"])
        xbr = float(p["xbr"])
        ybr = float(p["ybr"])
        if xbr <= xtl or ybr <= ytl:
            raise ValueError(
                f"Degenerate bbox (xtl={xtl}, ytl={ytl}, xbr={xbr}, ybr={ybr})"
            )
        boxes.append(AIhubBox(category="plant_roi", xyxy=(xtl, ytl, xbr, ybr)))

    return AIhubImage(
        image_path=image_path,
        crop=crop_name,
        width=width,
        height=height,
        disease_code=disease_code,
        fireblight=1 if disease_code != 0 else 0,
        boxes=tuple(boxes),
    )


def _match_image_path(
    ann_path: Path,
    images_dir: Path,
    image_exts: Iterable[str],
) -> Path | None:
    """`V006_..._0001_S01_1.jpg.json` → `V006_..._0001_S01_1.jpg` 위치 탐색.

    실제 AIhub 라벨 파일명은 `<image_basename>.json` 패턴(이미지 확장자 포함)이므로
    파일명에서 `.json` 만 떼면 이미지 파일명이 된다. 이미지 확장자가 `.jpg`·`.JPG` 등
    실제 파일과 대소문자 다를 수 있어 폴백 탐색을 수행.
    """
    candidate = images_dir / ann_path.name[: -len(".json")]
    if candidate.exists():
        return candidate
    stem = candidate.stem  # basename without final extension
    for ext in image_exts:
        alt = images_dir / f"{stem}{ext}"
        if alt.exists():
            return alt
    return None


def load_aihub_split(
    root: Path,
    *,
    image_exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".JPG", ".PNG"),
    annotation_loader: AnnotationLoader = _load_annotation_json,
    strict: bool = False,
) -> list[AIhubImage]:
    """`root/images/` + `root/annotations/` 쌍으로부터 항목 수집.

    라벨 파일 이름은 `<image_filename>.json` 형식을 가정한다 (AIhub 관례).
    즉 `annotations/V006_..._1.jpg.json` 은 `images/V006_..._1.jpg` 와 짝.

    Args:
        root: `images/` / `annotations/` 하위 디렉토리를 가진 루트.
        image_exts: 이미지 확장자 후보 (대소문자 혼재 대응).
        annotation_loader: JSON dict 로드 함수. 포맷 교체 대응.
        strict: True면 이미지 누락 시 FileNotFoundError; False면 조용히 skip.

    Raises:
        ValueError: 파싱 실패 (파일 경로 메시지에 포함).
        FileNotFoundError: strict=True + 이미지 누락.
    """
    images_dir = root / "images"
    ann_dir = root / "annotations"
    entries: list[AIhubImage] = []
    ann_paths = sorted(ann_dir.glob("*.json"))
    for ann_path in ann_paths:
        img_path = _match_image_path(ann_path, images_dir, image_exts)
        if img_path is None:
            if strict:
                raise FileNotFoundError(
                    f"이미지 매칭 실패: {ann_path} → {images_dir}"
                )
            continue
        ann = annotation_loader(ann_path)
        try:
            entries.append(_parse_entry(img_path, ann))
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"어노테이션 파싱 실패 ({ann_path}): {exc}") from exc
    return entries

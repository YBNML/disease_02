from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from disease_detection.data.aihub import AIhubImage, CROP_CODE_TO_NAME, load_aihub_split
from disease_detection.data.classification_dataset import (
    ClassificationCropDataset,
    MultiPartCropDataset,
    PART_STATES,
    PLANT_PARTS,
    build_defect_items,
    build_fireblight_items,
)
from disease_detection.data.detection_dataset import (
    DetectionDataset,
    NUM_CLASSES,
    PART_CATEGORIES,
)
from disease_detection.data.transforms import (
    build_classifier_eval_transform,
    build_classifier_train_transform,
    build_detector_eval_transform,
    build_detector_train_transform,
)


# ── transforms ──────────────────────────────────────────────────────────


def test_classifier_train_transform_output_shape():
    tfm = build_classifier_train_transform()
    img = torch.randint(0, 255, (3, 512, 640), dtype=torch.uint8)
    out = tfm(img)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_classifier_eval_transform_output_shape():
    tfm = build_classifier_eval_transform()
    img = torch.randint(0, 255, (3, 512, 640), dtype=torch.uint8)
    out = tfm(img)
    assert out.shape == (3, 224, 224)


def test_detector_train_transform_preserves_bbox_count():
    tfm = build_detector_train_transform()
    img = torch.randint(0, 255, (3, 400, 400), dtype=torch.uint8)
    from torchvision import tv_tensors

    boxes = tv_tensors.BoundingBoxes(
        [[10, 10, 100, 100], [50, 50, 200, 200]],
        format="XYXY",
        canvas_size=(400, 400),
    )
    target = {"boxes": boxes, "labels": torch.tensor([1, 2])}
    img_out, target_out = tfm(img, target)
    assert len(target_out["boxes"]) == 2
    assert img_out.dtype == torch.float32


# ── AIhub parser (real schema) ──────────────────────────────────────────


def test_crop_code_to_name_mapping():
    assert CROP_CODE_TO_NAME == {1: "pear", 2: "apple"}


def test_load_aihub_split_parses_two_samples(fixtures_dir):
    entries = load_aihub_split(fixtures_dir / "dummy_aihub")
    assert len(entries) == 2
    by_crop = {e.crop: e for e in entries}

    pear = by_crop["pear"]
    assert isinstance(pear, AIhubImage)
    assert pear.width == 640
    assert pear.height == 480
    assert pear.disease_code == 0
    assert pear.fireblight == 0
    assert len(pear.boxes) == 1
    assert pear.boxes[0].category == "plant_roi"
    assert pear.boxes[0].xyxy == (100.0, 80.0, 350.0, 380.0)

    apple = by_crop["apple"]
    assert apple.width == 800
    assert apple.height == 600
    assert apple.disease_code == 5
    assert apple.fireblight == 1  # disease_code != 0 → 이진 1
    assert len(apple.boxes) == 1


def test_load_aihub_split_uppercase_jpg_extension(fixtures_dir):
    """apple fixture 는 `.JPG` 확장자를 사용 — 대소문자 매칭이 OK인지 확인."""
    entries = load_aihub_split(fixtures_dir / "dummy_aihub")
    apple = next(e for e in entries if e.crop == "apple")
    assert apple.image_path.suffix == ".JPG"


def test_load_aihub_split_accepts_custom_loader(fixtures_dir):
    """커스텀 annotation_loader 주입 시 스키마 변환 가능함을 검증."""

    def loader(p: Path) -> dict:
        raw = json.loads(p.read_text(encoding="utf-8"))
        raw["annotations"]["disease"] = 9  # 임의로 덮어써 binary 1 되는지 확인
        return raw

    entries = load_aihub_split(fixtures_dir / "dummy_aihub", annotation_loader=loader)
    assert all(e.fireblight == 1 for e in entries)


def test_load_aihub_split_strict_raises_on_missing_image(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "annotations").mkdir()
    # 라벨만 있고 이미지 없음
    (tmp_path / "annotations" / "foo.jpg.json").write_text(
        json.dumps(
            {
                "description": {"image": "foo.jpg", "width": 10, "height": 10},
                "annotations": {
                    "disease": 0,
                    "crop": 1,
                    "points": [{"xtl": 0, "ytl": 0, "xbr": 5, "ybr": 5}],
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="foo.jpg.json"):
        load_aihub_split(tmp_path, strict=True)


def test_load_aihub_split_wraps_parse_errors_with_file_path(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "annotations").mkdir()
    (tmp_path / "images" / "bad.jpg").write_bytes(b"")
    # annotations 필드가 dict 대신 list → TypeError
    (tmp_path / "annotations" / "bad.jpg.json").write_text(
        json.dumps(
            {
                "description": {"image": "bad.jpg", "width": 10, "height": 10},
                "annotations": [{"disease": 0}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bad.jpg.json"):
        load_aihub_split(tmp_path)


# ── DetectionDataset (single-class) ─────────────────────────────────────


def test_part_categories_single_class():
    assert PART_CATEGORIES == {"plant_roi": 1}
    assert NUM_CLASSES == 2  # background + plant_roi


def test_detection_dataset_item_shapes(fixtures_dir):
    ds = DetectionDataset.from_aihub_root(fixtures_dir / "dummy_aihub")
    assert len(ds) == 2
    img, target = ds[0]
    assert img.ndim == 3 and img.shape[0] == 3  # CHW
    assert target["boxes"].shape == (1, 4)  # 이미지당 bbox 정확히 1개
    assert target["labels"].tolist() == [1]  # plant_roi
    assert target["image_id"].numel() == 1
    assert "area" in target and target["area"].shape == (1,)


# ── ClassificationCropDataset (binary fireblight) ───────────────────────


def test_build_fireblight_items_binarizes_disease_code(fixtures_dir):
    items = build_fireblight_items(fixtures_dir / "dummy_aihub")
    assert len(items) == 2  # 2 images × 1 bbox each
    by_crop = {it.crop: it for it in items}
    assert by_crop["pear"].label == 0   # disease_code=0
    assert by_crop["apple"].label == 1  # disease_code=5 → binary 1


def test_classification_crop_dataset_returns_tensor(fixtures_dir):
    items = build_fireblight_items(fixtures_dir / "dummy_aihub")
    ds = ClassificationCropDataset(items)
    img, label = ds[0]
    assert img.ndim == 3 and img.shape[0] == 3
    assert label in (0, 1)


# ── MultiPartCropDataset (4-part × 3-class) ──────────────────────────────


def test_plant_parts_and_states_constants():
    assert PLANT_PARTS == ("leaf", "branch", "fruit", "stem")
    assert PART_STATES == {"absent": 0, "normal": 1, "defect": 2}


def test_build_defect_items_parses_v2_jsonl(fixtures_dir):
    items = build_defect_items(
        fixtures_dir / "dummy_aihub",
        fixtures_dir / "dummy_vlm_labels.jsonl",
    )
    assert len(items) == 2
    by_crop = {it.crop: it for it in items}
    # pear fixture: leaf=defect, branch=normal, fruit=absent, stem=absent
    assert by_crop["pear"].part_states == (
        PART_STATES["defect"],
        PART_STATES["normal"],
        PART_STATES["absent"],
        PART_STATES["absent"],
    )
    # apple fixture: leaf=normal, branch=absent, fruit=defect, stem=absent
    assert by_crop["apple"].part_states == (
        PART_STATES["normal"],
        PART_STATES["absent"],
        PART_STATES["defect"],
        PART_STATES["absent"],
    )


def test_multipart_crop_dataset_returns_vector_label(fixtures_dir):
    items = build_defect_items(
        fixtures_dir / "dummy_aihub",
        fixtures_dir / "dummy_vlm_labels.jsonl",
    )
    ds = MultiPartCropDataset(items)
    img, y = ds[0]
    assert img.ndim == 3 and img.shape[0] == 3
    assert y.shape == (4,)
    assert y.dtype == torch.long


def test_load_aihub_split_rejects_degenerate_bbox(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "annotations").mkdir()
    (tmp_path / "images" / "deg.jpg").write_bytes(b"")
    (tmp_path / "annotations" / "deg.jpg.json").write_text(
        json.dumps(
            {
                "description": {"image": "deg.jpg", "width": 100, "height": 100},
                "annotations": {
                    "disease": 0,
                    "crop": 1,
                    # xbr <= xtl → 퇴화
                    "points": [{"xtl": 50, "ytl": 50, "xbr": 30, "ybr": 80}],
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Degenerate bbox"):
        load_aihub_split(tmp_path)

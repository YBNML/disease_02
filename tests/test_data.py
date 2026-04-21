from __future__ import annotations

from pathlib import Path

import torch

from disease_detection.data.aihub import AIhubImage, load_aihub_split
from disease_detection.data.classification_dataset import (
    build_fireblight_crops,
    build_defect_crops,
    ClassificationCropDataset,
)
from disease_detection.data.detection_dataset import DetectionDataset, PART_CATEGORIES
from disease_detection.data.transforms import (
    build_classifier_train_transform,
    build_classifier_eval_transform,
    build_detector_train_transform,
    build_detector_eval_transform,
)


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
    # torchvision v2 detection transforms use tv_tensors
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


def test_load_aihub_split_parses_fixture(fixtures_dir):
    root = fixtures_dir / "dummy_aihub"
    entries = load_aihub_split(root)
    assert len(entries) == 1
    entry = entries[0]
    assert isinstance(entry, AIhubImage)
    assert entry.crop == "pear"
    assert entry.fireblight == 1
    assert entry.width == 640
    assert entry.height == 480
    assert len(entry.boxes) == 2
    assert entry.boxes[0].category == "leaf"
    assert entry.boxes[0].xyxy == (100.0, 80.0, 350.0, 380.0)  # x+w, y+h


def test_load_aihub_split_accepts_custom_loader(fixtures_dir):
    """Pluggable loader: inject a function that normalizes alternate schemas."""
    root = fixtures_dir / "dummy_aihub"

    def loader(p: Path) -> dict:
        import json

        raw = json.loads(p.read_text(encoding="utf-8"))
        raw["meta"] = {"crop": "apple"}  # override via loader
        return raw

    entries = load_aihub_split(root, annotation_loader=loader)
    assert entries[0].crop == "apple"


def test_load_aihub_split_strict_raises_on_missing(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "annotations").mkdir()
    (tmp_path / "images" / "x.jpg").write_bytes(b"")
    import pytest

    with pytest.raises(FileNotFoundError, match="x.json"):
        load_aihub_split(tmp_path, strict=True)


def test_load_aihub_split_wraps_parse_errors_with_file_path(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "annotations").mkdir()
    (tmp_path / "images" / "bad.jpg").write_bytes(b"")
    (tmp_path / "annotations" / "bad.json").write_text(
        '{"image": {"width": 100, "height": 100}, "annotations": [{"bbox": [1,2,3], "category": "leaf"}]}'
    )
    import pytest

    with pytest.raises(ValueError, match="bad.json"):
        load_aihub_split(tmp_path)


def test_part_categories_mapping():
    # background=0, leaf=1, stem=2, fruit=3
    assert PART_CATEGORIES["leaf"] == 1
    assert PART_CATEGORIES["stem"] == 2
    assert PART_CATEGORIES["fruit"] == 3


def test_detection_dataset_item_shapes(fixtures_dir):
    ds = DetectionDataset.from_aihub_root(fixtures_dir / "dummy_aihub")
    img, target = ds[0]
    assert img.ndim == 3 and img.shape[0] == 3  # CHW
    assert target["boxes"].shape == (2, 4)
    assert target["labels"].tolist() == [1, 3]  # leaf, fruit
    assert target["image_id"].numel() == 1


def test_build_fireblight_crops_uses_aihub_labels(fixtures_dir):
    items = build_fireblight_crops(fixtures_dir / "dummy_aihub")
    # sample1 has fireblight=1, 2 boxes → 2 crop items, both label=1
    assert len(items) == 2
    assert all(item.label == 1 for item in items)
    assert {item.plant_part for item in items} == {"leaf", "fruit"}


def test_build_defect_crops_applies_severity_threshold(fixtures_dir):
    items = build_defect_crops(
        fixtures_dir / "dummy_aihub",
        fixtures_dir / "dummy_vlm_labels.jsonl",
        defect_threshold=4,
    )
    # severity=6 → defect (label=1). Only leaf box matches vlm plant_part.
    assert len(items) >= 1
    assert any(item.label == 1 and item.plant_part == "leaf" for item in items)


def test_classification_crop_dataset_returns_tensor(fixtures_dir):
    items = build_fireblight_crops(fixtures_dir / "dummy_aihub")
    ds = ClassificationCropDataset(items)
    img, label = ds[0]
    assert img.ndim == 3 and img.shape[0] == 3
    assert label in (0, 1)

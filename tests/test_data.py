from __future__ import annotations

from pathlib import Path

import torch

from disease_detection.data.aihub import AIhubImage, load_aihub_split
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

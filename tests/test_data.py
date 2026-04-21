from __future__ import annotations

import torch

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

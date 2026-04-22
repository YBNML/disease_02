"""torchvision v2 기반 변환 생성기.

분류기는 단일 텐서, detector는 이미지+target 쌍을 다룸. v2는 두 경우 모두 동일 API.
"""
from __future__ import annotations

import torch
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_classifier_train_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(224, scale=(0.7, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.RandomErasing(p=0.25, scale=(0.02, 0.1)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_classifier_eval_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256, antialias=True),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_detector_train_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def build_detector_eval_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

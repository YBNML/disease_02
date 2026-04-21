"""Detector + 두 분류기 결합 추론 래퍼.

Detector는 부위 bbox를 예측하고, 각 bbox를 crop하여 두 분류기 각각에 입력.
`fireblight_prob`과 `defect_prob`을 동시에 반환.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2

from ..data.detection_dataset import PART_CATEGORIES
from ..data.transforms import (
    build_classifier_eval_transform,
    build_detector_eval_transform,
)

_INV_PART = {v: k for k, v in PART_CATEGORIES.items()}


class _DetectorLike(Protocol):
    def __call__(self, images: list[torch.Tensor]) -> list[dict]: ...


class _ClassifierLike(Protocol):
    def __call__(self, crops: torch.Tensor) -> torch.Tensor: ...


@dataclass
class Detection:
    plant_part: str
    xyxy: tuple[float, float, float, float]
    score: float
    fireblight_prob: float
    defect_prob: float


@dataclass
class PipelinePrediction:
    image_path: Path
    crop: str
    detections: list[Detection] = field(default_factory=list)


class TwoStagePipeline:
    def __init__(
        self,
        detector: _DetectorLike,
        fireblight_classifier: _ClassifierLike,
        defect_classifier: _ClassifierLike,
        score_threshold: float = 0.5,
        detector_transform: v2.Compose | None = None,
        classifier_transform: v2.Compose | None = None,
    ) -> None:
        self.detector = detector
        self.fireblight = fireblight_classifier
        self.defect = defect_classifier
        self.score_threshold = score_threshold
        self.det_tfm = detector_transform or build_detector_eval_transform()
        self.cls_tfm = classifier_transform or build_classifier_eval_transform()

    def predict_image(self, image_path: Path, crop: str) -> PipelinePrediction:
        with Image.open(image_path) as src:
            pil = src.convert("RGB")
        det_input = self.det_tfm(tv_tensors.Image(pil))
        outputs = self.detector([det_input])[0]

        boxes = outputs["boxes"]
        labels = outputs["labels"]
        scores = outputs["scores"]

        keep = scores >= self.score_threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]

        detections: list[Detection] = []
        if len(boxes) == 0:
            return PipelinePrediction(image_path=image_path, crop=crop, detections=[])

        # crop + 분류기 배치 입력
        crops: list[torch.Tensor] = []
        for xyxy in boxes.tolist():
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
            c = pil.crop((x1, y1, x2, y2))
            crops.append(self.cls_tfm(tv_tensors.Image(c)))
        crop_batch = torch.stack(crops)

        fire_probs = torch.sigmoid(self.fireblight(crop_batch)).tolist()
        defect_probs = torch.sigmoid(self.defect(crop_batch)).tolist()

        for box, label_id, score, fp, dp in zip(
            boxes.tolist(), labels.tolist(), scores.tolist(), fire_probs, defect_probs
        ):
            detections.append(
                Detection(
                    plant_part=_INV_PART.get(int(label_id), "unknown"),
                    xyxy=tuple(float(v) for v in box),
                    score=float(score),
                    fireblight_prob=float(fp),
                    defect_prob=float(dp),
                )
            )
        return PipelinePrediction(
            image_path=image_path, crop=crop, detections=detections
        )

"""Detector + 두 분류기 결합 추론 래퍼.

단계:
1. Faster R-CNN single-class detector 가 plant ROI bbox 예측
2. 각 bbox 를 crop → 두 분류기 병렬 실행
   - `fireblight_classifier`: (B,) 이진 logit → sigmoid → `fireblight_prob`
   - `defect_classifier` (multi-part): (B, 4, 3) logit → softmax (dim=-1) →
     각 부위 `state=defect` 확률을 `part_defect_probs[part]` 에 저장

배경 clamp·빈 bbox 처리·autograd 비활성화는 Phase 1 그대로.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2

from ..data.classification_dataset import PART_STATES, PLANT_PARTS
from ..data.detection_dataset import PART_CATEGORIES
from ..data.transforms import (
    build_classifier_eval_transform,
    build_detector_eval_transform,
)

_INV_PART = {v: k for k, v in PART_CATEGORIES.items()}
_DEFECT_STATE_INDEX = PART_STATES["defect"]


class _DetectorLike(Protocol):
    def __call__(self, images: list[torch.Tensor]) -> list[dict]: ...


class _FireblightClassifierLike(Protocol):
    def __call__(self, crops: torch.Tensor) -> torch.Tensor:
        """Return shape `(B,)` binary logits."""


class _DefectClassifierLike(Protocol):
    def __call__(self, crops: torch.Tensor) -> torch.Tensor:
        """Return shape `(B, NUM_PARTS, NUM_STATES)` logits for multi-part head."""


@dataclass
class Detection:
    roi_category: str            # detector label, 현재는 항상 "plant_roi"
    xyxy: tuple[float, float, float, float]
    score: float                 # detector confidence
    fireblight_prob: float
    part_defect_probs: dict[str, float]   # {leaf, branch, fruit, stem} → P(state=defect)


@dataclass
class PipelinePrediction:
    image_path: Path
    crop: str
    detections: list[Detection] = field(default_factory=list)


class TwoStagePipeline:
    def __init__(
        self,
        detector: _DetectorLike,
        fireblight_classifier: _FireblightClassifierLike,
        defect_classifier: _DefectClassifierLike,
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

    @torch.inference_mode()
    def predict_image(self, image_path: Path, crop: str) -> PipelinePrediction:
        with Image.open(image_path) as src:
            pil = src.convert("RGB")
        width, height = pil.size
        det_input = self.det_tfm(tv_tensors.Image(pil))
        outputs = self.detector([det_input])[0]

        boxes = outputs["boxes"]
        labels = outputs["labels"]
        scores = outputs["scores"]

        keep = scores >= self.score_threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]

        # 이미지 경계 밖 bbox 는 PIL 이 검은색 패딩 → 분류기 입력 오염. clamp 필수.
        if len(boxes) > 0:
            boxes_clamped = boxes.clone()
            boxes_clamped[:, 0].clamp_(0, width)
            boxes_clamped[:, 1].clamp_(0, height)
            boxes_clamped[:, 2].clamp_(0, width)
            boxes_clamped[:, 3].clamp_(0, height)
            valid = (boxes_clamped[:, 2] > boxes_clamped[:, 0]) & (
                boxes_clamped[:, 3] > boxes_clamped[:, 1]
            )
            boxes = boxes_clamped[valid]
            labels = labels[valid]
            scores = scores[valid]

        if len(boxes) == 0:
            return PipelinePrediction(image_path=image_path, crop=crop, detections=[])

        crops: list[torch.Tensor] = []
        for xyxy in boxes.tolist():
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
            c = pil.crop((x1, y1, x2, y2))
            crops.append(self.cls_tfm(tv_tensors.Image(c)))
        crop_batch = torch.stack(crops)

        # Fireblight 분류기: (B,) 이진 logit 기대
        fire_logits = self.fireblight(crop_batch)
        if fire_logits.ndim != 1 or fire_logits.shape[0] != crop_batch.shape[0]:
            raise ValueError(
                f"fireblight classifier returned shape {tuple(fire_logits.shape)}, "
                f"expected ({crop_batch.shape[0]},)"
            )
        fire_probs = torch.sigmoid(fire_logits).tolist()

        # Defect 분류기: (B, NUM_PARTS, NUM_STATES) — multi-part head
        defect_logits = self.defect(crop_batch)
        if (
            defect_logits.ndim != 3
            or defect_logits.shape[0] != crop_batch.shape[0]
            or defect_logits.shape[1] != len(PLANT_PARTS)
            or defect_logits.shape[2] != len(PART_STATES)
        ):
            raise ValueError(
                f"defect classifier returned shape {tuple(defect_logits.shape)}, "
                f"expected ({crop_batch.shape[0]}, {len(PLANT_PARTS)}, {len(PART_STATES)})"
            )
        defect_softmax = torch.softmax(defect_logits, dim=-1)
        defect_probs = defect_softmax[..., _DEFECT_STATE_INDEX].tolist()

        detections: list[Detection] = []
        for i, (box, label_id, score, fp) in enumerate(
            zip(boxes.tolist(), labels.tolist(), scores.tolist(), fire_probs)
        ):
            part_probs = {
                part_name: float(defect_probs[i][j])
                for j, part_name in enumerate(PLANT_PARTS)
            }
            detections.append(
                Detection(
                    roi_category=_INV_PART.get(int(label_id), "unknown"),
                    xyxy=tuple(float(v) for v in box),
                    score=float(score),
                    fireblight_prob=float(fp),
                    part_defect_probs=part_probs,
                )
            )
        return PipelinePrediction(
            image_path=image_path, crop=crop, detections=detections
        )

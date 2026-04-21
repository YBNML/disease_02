"""End-to-end 평가 경로.

- `evaluate_classifier_oracle` — GT bbox로 crop한 분류 입력에 대한 분류기 단독 평가 (상한).
- `evaluate_pipeline_realistic_image` — Detector 예측 bbox로 이미지 단위 화상병 집계.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import DataLoader

from ..data.aihub import AIhubImage
from ..data.classification_dataset import ClassificationCropDataset, CropItem
from ..data.transforms import build_classifier_eval_transform
from ..models.pipeline import TwoStagePipeline
from .metrics import ClassificationMetricsReport, compute_classification_report


def _collate_tensor_labels(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys


def evaluate_classifier_oracle(
    items: list[CropItem],
    classifier_fn: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int = 16,
    device: str | torch.device = "cpu",
    threshold: float = 0.5,
) -> ClassificationMetricsReport:
    """GT bbox crop으로 분류기 단독 성능 측정.

    `classifier_fn`은 이미 eval 모드로 준비된 상태여야 한다 (BN/Dropout 등의
    모드 전환은 호출자 책임. 이 함수는 autograd 비활성화만 담당).
    """
    ds = ClassificationCropDataset(items, transform=build_classifier_eval_transform())
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=_collate_tensor_labels)

    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.inference_mode():
        for xs, ys in loader:
            xs = xs.to(device)
            logits = classifier_fn(xs)
            all_scores.append(torch.sigmoid(logits.detach().cpu()))
            all_labels.append(ys)

    y_score = torch.cat(all_scores)
    y_true = torch.cat(all_labels)
    return compute_classification_report(y_true, y_score, threshold=threshold)


def evaluate_pipeline_realistic_image(
    entries: list[AIhubImage],
    pipeline: TwoStagePipeline,
    threshold: float = 0.5,
) -> ClassificationMetricsReport:
    """이미지 단위 화상병 분류 (realistic path).

    각 이미지에 대해 파이프라인이 예측한 detection 중 최대 `fireblight_prob`을
    이미지 스코어로 집계. 예측된 detection이 없으면 0.0. GT는 AIhub 원본 이미지
    단위 `fireblight` 라벨. Oracle(crop 단위)과 비교 가능한 형태로 이미지 단위
    스코어를 별도 리포트.
    """
    scores: list[float] = []
    labels: list[int] = []
    for entry in entries:
        pred = pipeline.predict_image(entry.image_path, crop=entry.crop)
        score = max((d.fireblight_prob for d in pred.detections), default=0.0)
        scores.append(score)
        labels.append(int(entry.fireblight))
    y_score = torch.tensor(scores)
    y_true = torch.tensor(labels)
    return compute_classification_report(y_true, y_score, threshold=threshold)

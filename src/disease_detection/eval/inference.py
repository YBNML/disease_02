"""End-to-end 평가 경로.

- `evaluate_classifier_oracle` — GT bbox로 crop한 분류 입력에 대한 분류기 단독 평가 (상한).
- `evaluate_pipeline_realistic` — Detector 예측 bbox를 활용한 2단계 파이프라인 평가.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import DataLoader

from ..data.classification_dataset import ClassificationCropDataset, CropItem
from ..data.transforms import build_classifier_eval_transform
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

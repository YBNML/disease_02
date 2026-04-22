"""torchmetrics 기반 classifier·detector 평가 리포트."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryPrecision,
    BinaryPrecisionRecallCurve,
    BinaryRecall,
)
from torchmetrics.detection import MeanAveragePrecision


@dataclass
class ClassificationMetricsReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    recall_at_precision_70: float | None
    confusion_matrix: list[list[int]]


def compute_classification_report(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    threshold: float = 0.5,
    target_precision: float = 0.7,
) -> ClassificationMetricsReport:
    y_true = y_true.long()
    preds = (y_score >= threshold).long()

    acc = float(BinaryAccuracy()(preds, y_true))
    prec = float(BinaryPrecision()(preds, y_true))
    rec = float(BinaryRecall()(preds, y_true))
    f1 = float(BinaryF1Score()(preds, y_true))
    cm = BinaryConfusionMatrix()(preds, y_true).tolist()

    try:
        roc_auc = float(BinaryAUROC()(y_score, y_true))
    except Exception:
        roc_auc = None

    # Recall@Precision>=target_precision
    pr = BinaryPrecisionRecallCurve()
    pr.update(y_score, y_true)
    precisions, recalls, _ = pr.compute()
    mask = precisions >= target_precision
    recall_at_p = float(recalls[mask].max()) if mask.any() else None

    return ClassificationMetricsReport(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc_auc,
        recall_at_precision_70=recall_at_p,
        confusion_matrix=cm,
    )


def compute_detection_map(
    preds: list[dict],
    targets: list[dict],
) -> dict[str, float]:
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=False)
    metric.update(preds, targets)
    raw = metric.compute()
    return {
        "map": float(raw["map"]),
        "map_50": float(raw["map_50"]),
        "map_75": float(raw["map_75"]),
    }

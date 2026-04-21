from __future__ import annotations

from pathlib import Path

import torch

from disease_detection.eval.inference import evaluate_classifier_oracle
from disease_detection.eval.metrics import (
    ClassificationMetricsReport,
    compute_classification_report,
    compute_detection_map,
)


def test_compute_classification_report_perfect():
    y_true = torch.tensor([0, 1, 1, 0, 1])
    y_score = torch.tensor([0.1, 0.9, 0.8, 0.2, 0.95])
    rep = compute_classification_report(y_true, y_score, threshold=0.5)
    assert isinstance(rep, ClassificationMetricsReport)
    assert rep.accuracy == 1.0
    assert rep.recall == 1.0
    assert rep.precision == 1.0
    assert rep.f1 == 1.0


def test_compute_classification_report_recall_at_precision():
    # Scores designed so that tuning threshold matters.
    y_true = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    y_score = torch.tensor([0.1, 0.2, 0.6, 0.4, 0.55, 0.7, 0.9])
    rep = compute_classification_report(y_true, y_score, threshold=0.5)
    # Recall@Precision>=0.7 should be defined (not None) for this distribution.
    assert rep.recall_at_precision_70 is not None


def test_compute_detection_map_smoke():
    preds = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.9]),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([1]),
        }
    ]
    m = compute_detection_map(preds, targets)
    assert "map_50" in m
    assert m["map_50"] >= 0.99


def test_evaluate_classifier_oracle_returns_report(fixtures_dir):
    from disease_detection.data.classification_dataset import build_fireblight_crops

    items = build_fireblight_crops(fixtures_dir / "dummy_aihub")

    def fake_classifier_fn(batch):
        # Always predict positive with high probability
        return torch.full((batch.shape[0],), 3.0)  # sigmoid(3)=0.95

    rep = evaluate_classifier_oracle(
        items,
        classifier_fn=fake_classifier_fn,
        batch_size=2,
    )
    assert rep.accuracy >= 0.0
    assert rep.recall == 1.0  # all-positive predictor catches all positives

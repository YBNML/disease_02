from __future__ import annotations

import torch
from PIL import Image

from disease_detection.models.pipeline import TwoStagePipeline, PipelinePrediction


class _FakeDetector:
    def __call__(self, images):
        return [
            {
                "boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0]]),
                "labels": torch.tensor([1]),  # leaf
                "scores": torch.tensor([0.9]),
            }
        ]


class _FakeClassifier:
    def __init__(self, prob: float) -> None:
        self.prob = prob

    def __call__(self, crops):
        batch = crops.shape[0]
        return torch.logit(torch.full((batch,), self.prob))


def test_pipeline_runs_end_to_end(tmp_path):
    img = Image.new("RGB", (200, 200), color=(200, 50, 50))
    img_path = tmp_path / "x.jpg"
    img.save(img_path)

    pipe = TwoStagePipeline(
        detector=_FakeDetector(),
        fireblight_classifier=_FakeClassifier(prob=0.8),
        defect_classifier=_FakeClassifier(prob=0.3),
        score_threshold=0.5,
    )
    result = pipe.predict_image(img_path, crop="pear")
    assert isinstance(result, PipelinePrediction)
    assert len(result.detections) == 1
    det = result.detections[0]
    assert det.plant_part == "leaf"
    assert det.fireblight_prob > 0.5
    assert det.defect_prob < 0.5


def test_pipeline_filters_by_score_threshold(tmp_path):
    class LowScoreDetector:
        def __call__(self, images):
            return [
                {
                    "boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0]]),
                    "labels": torch.tensor([1]),
                    "scores": torch.tensor([0.1]),
                }
            ]

    img_path = tmp_path / "x.jpg"
    Image.new("RGB", (120, 120)).save(img_path)

    pipe = TwoStagePipeline(
        detector=LowScoreDetector(),
        fireblight_classifier=_FakeClassifier(prob=0.9),
        defect_classifier=_FakeClassifier(prob=0.9),
        score_threshold=0.5,
    )
    result = pipe.predict_image(img_path, crop="pear")
    assert result.detections == []

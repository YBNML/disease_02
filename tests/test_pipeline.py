from __future__ import annotations

import pytest
import torch
from PIL import Image

from disease_detection.data.classification_dataset import PART_STATES, PLANT_PARTS
from disease_detection.models.pipeline import PipelinePrediction, TwoStagePipeline


class _FakeDetector:
    def __call__(self, images):
        return [
            {
                "boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0]]),
                "labels": torch.tensor([1]),  # plant_roi
                "scores": torch.tensor([0.9]),
            }
        ]


class _FakeFireblight:
    def __init__(self, prob: float) -> None:
        self.prob = prob

    def __call__(self, crops):
        B = crops.shape[0]
        return torch.logit(torch.full((B,), self.prob))


class _FakeMultiPart:
    """각 (부위, state) 에 대한 원-핫 logit을 생성. state 는 `PART_STATES` 인덱스."""

    def __init__(self, per_part_state: dict[str, str]) -> None:
        self._per_part_state = per_part_state

    def __call__(self, crops):
        B = crops.shape[0]
        logits = torch.full((B, len(PLANT_PARTS), len(PART_STATES)), -10.0)
        for i, part in enumerate(PLANT_PARTS):
            state_idx = PART_STATES[self._per_part_state[part]]
            logits[:, i, state_idx] = 10.0
        return logits


def test_pipeline_runs_end_to_end(tmp_path):
    img_path = tmp_path / "x.jpg"
    Image.new("RGB", (200, 200), color=(200, 50, 50)).save(img_path)

    pipe = TwoStagePipeline(
        detector=_FakeDetector(),
        fireblight_classifier=_FakeFireblight(prob=0.8),
        defect_classifier=_FakeMultiPart(
            per_part_state={
                "leaf": "defect",
                "branch": "normal",
                "fruit": "absent",
                "stem": "absent",
            }
        ),
        score_threshold=0.5,
    )
    result = pipe.predict_image(img_path, crop="pear")
    assert isinstance(result, PipelinePrediction)
    assert len(result.detections) == 1
    det = result.detections[0]
    assert det.roi_category == "plant_roi"
    assert det.fireblight_prob > 0.5
    # leaf만 defect → leaf prob 높고, 나머지는 낮음
    assert det.part_defect_probs["leaf"] > 0.9
    assert det.part_defect_probs["branch"] < 0.1
    assert det.part_defect_probs["fruit"] < 0.1
    assert det.part_defect_probs["stem"] < 0.1


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
        fireblight_classifier=_FakeFireblight(prob=0.9),
        defect_classifier=_FakeMultiPart(
            per_part_state={p: "absent" for p in PLANT_PARTS}
        ),
        score_threshold=0.5,
    )
    result = pipe.predict_image(img_path, crop="pear")
    assert result.detections == []


def test_pipeline_clamps_out_of_range_bboxes(tmp_path):
    class OutOfRangeDetector:
        def __call__(self, images):
            return [
                {
                    "boxes": torch.tensor([[-10.0, -10.0, 150.0, 150.0]]),
                    "labels": torch.tensor([1]),
                    "scores": torch.tensor([0.9]),
                }
            ]

    img_path = tmp_path / "x.jpg"
    Image.new("RGB", (100, 100)).save(img_path)
    pipe = TwoStagePipeline(
        detector=OutOfRangeDetector(),
        fireblight_classifier=_FakeFireblight(prob=0.5),
        defect_classifier=_FakeMultiPart(
            per_part_state={p: "absent" for p in PLANT_PARTS}
        ),
        score_threshold=0.5,
    )
    result = pipe.predict_image(img_path, crop="pear")
    assert len(result.detections) == 1
    x1, y1, x2, y2 = result.detections[0].xyxy
    assert 0 <= x1 and 0 <= y1
    assert x2 <= 100 and y2 <= 100


def test_pipeline_drops_degenerate_bboxes(tmp_path):
    class DegenerateDetector:
        def __call__(self, images):
            return [
                {
                    "boxes": torch.tensor([[50.0, 50.0, 50.0, 50.0]]),
                    "labels": torch.tensor([1]),
                    "scores": torch.tensor([0.9]),
                }
            ]

    img_path = tmp_path / "x.jpg"
    Image.new("RGB", (100, 100)).save(img_path)
    pipe = TwoStagePipeline(
        detector=DegenerateDetector(),
        fireblight_classifier=_FakeFireblight(prob=0.5),
        defect_classifier=_FakeMultiPart(
            per_part_state={p: "absent" for p in PLANT_PARTS}
        ),
        score_threshold=0.5,
    )
    result = pipe.predict_image(img_path, crop="pear")
    assert result.detections == []


def test_pipeline_rejects_wrong_fireblight_shape(tmp_path):
    class TwoColFireblight:
        def __call__(self, crops):
            return torch.zeros(crops.shape[0], 2)

    img_path = tmp_path / "x.jpg"
    Image.new("RGB", (100, 100)).save(img_path)
    pipe = TwoStagePipeline(
        detector=_FakeDetector(),
        fireblight_classifier=TwoColFireblight(),
        defect_classifier=_FakeMultiPart(
            per_part_state={p: "absent" for p in PLANT_PARTS}
        ),
        score_threshold=0.5,
    )
    with pytest.raises(ValueError, match="fireblight"):
        pipe.predict_image(img_path, crop="pear")


def test_pipeline_rejects_wrong_defect_shape(tmp_path):
    class WrongDefect:
        def __call__(self, crops):
            return torch.zeros(crops.shape[0], 2)  # 2D — should be 3D

    img_path = tmp_path / "x.jpg"
    Image.new("RGB", (100, 100)).save(img_path)
    pipe = TwoStagePipeline(
        detector=_FakeDetector(),
        fireblight_classifier=_FakeFireblight(prob=0.5),
        defect_classifier=WrongDefect(),
        score_threshold=0.5,
    )
    with pytest.raises(ValueError, match="defect"):
        pipe.predict_image(img_path, crop="pear")

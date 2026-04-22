from __future__ import annotations

import torch

from disease_detection.data.detection_dataset import NUM_CLASSES as DETECTOR_NUM_CLASSES
from disease_detection.models.classifier import PlantDefectClassifier
from disease_detection.models.detector import FasterRCNNModule
from disease_detection.models.multipart_classifier import (
    NUM_PARTS,
    NUM_STATES,
    MultiPartDefectClassifier,
)


def test_detector_num_classes_matches_dataset():
    """설정 (FasterRCNNModule.num_classes) 과 dataset (NUM_CLASSES) 동기화."""
    assert DETECTOR_NUM_CLASSES == 2  # background + plant_roi


def test_detector_module_forward_smoke():
    module = FasterRCNNModule(num_classes=DETECTOR_NUM_CLASSES)
    module.eval()
    images = [torch.rand(3, 320, 320)]
    with torch.no_grad():
        outputs = module(images)
    assert len(outputs) == 1
    assert set(outputs[0].keys()) >= {"boxes", "labels", "scores"}


def test_detector_training_step_returns_loss():
    module = FasterRCNNModule(num_classes=DETECTOR_NUM_CLASSES)
    module.train()
    images = [torch.rand(3, 256, 256)]
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0]]),
            "labels": torch.tensor([1]),
        }
    ]
    loss = module.training_step((images, targets), batch_idx=0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_classifier_forward_shape():
    module = PlantDefectClassifier()
    module.eval()
    x = torch.rand(4, 3, 224, 224)
    with torch.no_grad():
        logits = module(x)
    assert logits.shape == (4,)


def test_classifier_training_step_returns_loss():
    module = PlantDefectClassifier()
    x = torch.rand(2, 3, 224, 224)
    y = torch.tensor([0, 1])
    loss = module.training_step((x, y), batch_idx=0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_classifier_overfits_two_samples():
    """10-step overfit sanity check — loss must decrease."""
    torch.manual_seed(0)
    module = PlantDefectClassifier(lr=1e-2)
    x = torch.rand(2, 3, 224, 224)
    y = torch.tensor([0, 1])
    opt = module.configure_optimizers()["optimizer"]
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = module.training_step((x, y), batch_idx=0)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] * 0.5


# ── MultiPartDefectClassifier (4 parts × 3 states) ──────────────────────


def test_multipart_constants():
    assert NUM_PARTS == 4
    assert NUM_STATES == 3


def test_multipart_forward_shape():
    module = MultiPartDefectClassifier()
    module.eval()
    x = torch.rand(3, 3, 224, 224)
    with torch.no_grad():
        logits = module(x)
    assert logits.shape == (3, NUM_PARTS, NUM_STATES)


def test_multipart_training_step_returns_loss():
    module = MultiPartDefectClassifier()
    x = torch.rand(2, 3, 224, 224)
    y = torch.tensor([[0, 1, 2, 0], [2, 0, 1, 2]], dtype=torch.long)
    loss = module.training_step((x, y), batch_idx=0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_multipart_overfits_two_samples():
    """2-sample multi-part overfit: 최종 loss가 최초의 절반 미만으로 떨어져야 한다."""
    torch.manual_seed(0)
    module = MultiPartDefectClassifier(lr=1e-2)
    x = torch.rand(2, 3, 224, 224)
    y = torch.tensor([[0, 1, 2, 0], [2, 0, 1, 2]], dtype=torch.long)
    opt = module.configure_optimizers()["optimizer"]
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = module.training_step((x, y), batch_idx=0)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.5

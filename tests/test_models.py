from __future__ import annotations

import torch

from disease_detection.models.classifier import PlantDefectClassifier
from disease_detection.models.detector import FasterRCNNModule


def test_detector_module_forward_smoke():
    module = FasterRCNNModule(num_classes=4)
    module.eval()
    images = [torch.rand(3, 320, 320)]
    with torch.no_grad():
        outputs = module(images)
    assert len(outputs) == 1
    assert set(outputs[0].keys()) >= {"boxes", "labels", "scores"}


def test_detector_training_step_returns_loss():
    module = FasterRCNNModule(num_classes=4)
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

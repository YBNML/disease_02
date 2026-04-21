from __future__ import annotations

import torch

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

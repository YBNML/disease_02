"""재현성을 위한 일괄 seed 설정."""
from __future__ import annotations

import pytorch_lightning as pl


def set_seed(seed: int) -> None:
    """Python / numpy / torch / PYTHONHASHSEED + DataLoader worker 시드를 일괄 설정.

    내부적으로 `pytorch_lightning.seed_everything(..., workers=True)`에 위임.
    `workers=True`는 `DataLoader`의 각 worker 프로세스에 결정적 시드를 배포한다
    (Phase 1의 detector·classifier 학습에서 multi-worker loading이 쓰이므로 중요).
    """
    pl.seed_everything(seed, workers=True)

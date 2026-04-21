"""재현성을 위한 일괄 seed 설정."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Python random / numpy / torch / PYTHONHASHSEED 동시 시드.

    Lightning의 `seed_everything`과 같은 역할을 하지만 의존성 최소화를 위해 직접 구현.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

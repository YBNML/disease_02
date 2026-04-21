from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from disease_detection.utils.io import resolve_dataset_root
from disease_detection.utils.seeding import set_seed


def test_resolve_dataset_root_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET_ROOT", str(tmp_path))
    assert resolve_dataset_root() == tmp_path


def test_resolve_dataset_root_raises_when_unset(monkeypatch):
    monkeypatch.delenv("DATASET_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="DATASET_ROOT"):
        resolve_dataset_root()


def test_resolve_dataset_root_raises_when_empty(monkeypatch):
    monkeypatch.setenv("DATASET_ROOT", "")
    with pytest.raises(RuntimeError, match="DATASET_ROOT"):
        resolve_dataset_root()


def test_resolve_dataset_root_raises_when_whitespace(monkeypatch):
    monkeypatch.setenv("DATASET_ROOT", "   ")
    with pytest.raises(RuntimeError, match="DATASET_ROOT"):
        resolve_dataset_root()


def test_resolve_dataset_root_expands_user(monkeypatch):
    monkeypatch.setenv("DATASET_ROOT", "~/nonexistent_ds_root_xyz")
    result = resolve_dataset_root(require_exists=False)
    assert result == Path("~/nonexistent_ds_root_xyz").expanduser()


def test_resolve_dataset_root_require_exists(monkeypatch):
    monkeypatch.setenv("DATASET_ROOT", "/definitely/does/not/exist/xyz")
    with pytest.raises(FileNotFoundError):
        resolve_dataset_root(require_exists=True)


def test_set_seed_makes_random_deterministic():
    set_seed(123)
    a = [random.random(), np.random.rand(), float(torch.rand(1))]
    set_seed(123)
    b = [random.random(), np.random.rand(), float(torch.rand(1))]
    assert a == b


def test_set_seed_different_values_differ():
    set_seed(1)
    a = float(torch.rand(1))
    set_seed(2)
    b = float(torch.rand(1))
    assert a != b

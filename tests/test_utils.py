from __future__ import annotations

import os
from pathlib import Path

import pytest

from disease_detection.utils.io import resolve_dataset_root


def test_resolve_dataset_root_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET_ROOT", str(tmp_path))
    assert resolve_dataset_root() == tmp_path


def test_resolve_dataset_root_raises_when_unset(monkeypatch):
    monkeypatch.delenv("DATASET_ROOT", raising=False)
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

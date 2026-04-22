"""scripts/ 진입점 유틸 테스트."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _load_module(name: str, filename: str):
    """Load a standalone script as a module for unit testing."""
    path = _SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── extract_aihub.parse_zip_filename ────────────────────────────────────


@pytest.fixture(scope="module")
def extract_aihub_mod():
    return _load_module("_test_extract_aihub", "extract_aihub.py")


@pytest.mark.parametrize(
    "name, expected_crop, expected_kind",
    [
        ("[라벨]배_0.정상.zip", "pear", "annotations"),
        ("[라벨]배_1.질병.zip", "pear", "annotations"),
        ("[라벨]사과_0.정상.zip", "apple", "annotations"),
        ("[라벨]사과_1.질병.zip", "apple", "annotations"),
        ("[원천]배_0.정상_(1).zip", "pear", "images"),
        ("[원천]배_0.정상_(2).zip", "pear", "images"),
        ("[원천]사과_1.질병.zip", "apple", "images"),
    ],
)
def test_parse_zip_filename_valid(extract_aihub_mod, name, expected_crop, expected_kind):
    job = extract_aihub_mod.parse_zip_filename(Path(name))
    assert job is not None
    assert job.crop == expected_crop
    assert job.kind == expected_kind


@pytest.mark.parametrize(
    "name",
    [
        # Multi-part 다운로드 fragment — 추출 대상이 아니라 스킵되어야 함.
        "[원천]사과_0.정상_(2).zip.irx577",
        # 알 수 없는 작물.
        "[라벨]복숭아_0.정상.zip",
        # tag 없음.
        "배_0.정상.zip",
        # 이상한 접미사.
        "[라벨]배_9.unknown.zip",
    ],
)
def test_parse_zip_filename_invalid(extract_aihub_mod, name):
    assert extract_aihub_mod.parse_zip_filename(Path(name)) is None


def test_collect_jobs_respects_labels_only(extract_aihub_mod, tmp_path):
    (tmp_path / "Training").mkdir()
    (tmp_path / "Validation").mkdir()
    for filename in (
        "[라벨]배_0.정상.zip",
        "[원천]배_0.정상_(1).zip",
    ):
        (tmp_path / "Training" / filename).write_bytes(b"")
    (tmp_path / "Validation" / "[라벨]사과_1.질병.zip").write_bytes(b"")

    all_jobs = extract_aihub_mod.collect_jobs(tmp_path)
    assert len(all_jobs) == 3

    labels_jobs = extract_aihub_mod.collect_jobs(tmp_path, labels_only=True)
    assert len(labels_jobs) == 2
    assert all(j.kind == "annotations" for j in labels_jobs)


def test_collect_jobs_skips_irx577_fragments(extract_aihub_mod, tmp_path):
    (tmp_path / "Training").mkdir()
    (tmp_path / "Training" / "[원천]사과_0.정상_(2).zip").write_bytes(b"")
    (tmp_path / "Training" / "[원천]사과_0.정상_(2).zip.irx577").write_bytes(b"")
    jobs = extract_aihub_mod.collect_jobs(tmp_path)
    assert len(jobs) == 1
    assert jobs[0].zip_path.name.endswith(".zip")

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from disease_detection.labeling.batch_label import (
    BatchJob,
    BatchResult,
    hash_image_file,
    load_completed_hashes,
    run_batch,
)
from disease_detection.labeling.prompts import SEVERITY_PROMPT_V1, PROMPT_VERSION
from disease_detection.labeling.vlm_client import (
    VLMLabel,
    call_claude_cli,
    parse_vlm_response,
)


def test_prompt_contains_pptx_keywords():
    # Keys from PPT Slide 2
    assert "NORMAL" in SEVERITY_PROMPT_V1
    assert "DEFECT" in SEVERITY_PROMPT_V1
    assert "severity" in SEVERITY_PROMPT_V1.lower()


def test_prompt_asks_for_structured_format():
    assert "Classification" in SEVERITY_PROMPT_V1
    assert "Severity" in SEVERITY_PROMPT_V1


def test_prompt_version_string():
    assert PROMPT_VERSION == "v1"


def test_parse_vlm_response_valid_json():
    raw = '{"classification": "DEFECT", "severity": 6, "explanation": "browning"}'
    label = parse_vlm_response(raw)
    assert label == VLMLabel(classification="DEFECT", severity=6, explanation="browning")


def test_parse_vlm_response_with_code_fences():
    raw = '```json\n{"classification": "NORMAL", "severity": 0, "explanation": "ok"}\n```'
    label = parse_vlm_response(raw)
    assert label.severity == 0


def test_parse_vlm_response_invalid_raises():
    import pytest

    with pytest.raises(ValueError):
        parse_vlm_response("not json at all")


def test_parse_vlm_response_handles_nested_objects():
    raw = (
        'Sure, here is the answer: '
        '{"classification": "DEFECT", "severity": 5, '
        '"meta": {"confidence": 0.9, "nested": {"k": 1}}, '
        '"explanation": "spot"}'
    )
    label = parse_vlm_response(raw)
    assert label.severity == 5
    assert label.classification == "DEFECT"


def test_parse_vlm_response_handles_braces_inside_strings():
    raw = '{"classification": "NORMAL", "severity": 0, "explanation": "text with } brace"}'
    label = parse_vlm_response(raw)
    assert label.explanation == "text with } brace"


def test_call_claude_cli_invokes_subprocess(mocker, tmp_path):
    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = '{"classification": "DEFECT", "severity": 5, "explanation": "x"}'
    completed.stderr = ""
    run_mock = mocker.patch(
        "disease_detection.labeling.vlm_client.subprocess.run",
        return_value=completed,
    )

    img = tmp_path / "a.jpg"
    img.write_bytes(b"fake")

    label = call_claude_cli(img, prompt="P", model="haiku")

    assert label.severity == 5
    args, kwargs = run_mock.call_args
    assert args[0][0] == "claude"
    assert "-p" in args[0]
    assert "--model" in args[0]
    assert "haiku" in args[0]
    assert str(img) in " ".join(args[0])


def test_call_claude_cli_nonzero_exit_raises(mocker, tmp_path):
    import pytest
    completed = MagicMock(returncode=1, stdout="", stderr="rate limit")
    mocker.patch(
        "disease_detection.labeling.vlm_client.subprocess.run",
        return_value=completed,
    )
    img = tmp_path / "a.jpg"
    img.write_bytes(b"fake")
    with pytest.raises(RuntimeError, match="rate limit"):
        call_claude_cli(img, prompt="P", model="haiku")


def test_call_claude_cli_missing_image_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        call_claude_cli(tmp_path / "nope.jpg", prompt="P", model="haiku")


def test_call_claude_cli_timeout_normalized_to_runtimeerror(mocker, tmp_path):
    import pytest

    mocker.patch(
        "disease_detection.labeling.vlm_client.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=1),
    )
    img = tmp_path / "a.jpg"
    img.write_bytes(b"fake")
    with pytest.raises(RuntimeError, match="timed out"):
        call_claude_cli(img, prompt="P", model="haiku", timeout_seconds=1)


def test_hash_image_file_deterministic(tmp_path):
    p = tmp_path / "x.jpg"
    p.write_bytes(b"hello world")
    h1 = hash_image_file(p)
    h2 = hash_image_file(p)
    assert h1 == h2 and len(h1) == 64  # sha256 hex


def test_load_completed_hashes_reads_existing_jsonl(tmp_path):
    jsonl = tmp_path / "labels.jsonl"
    jsonl.write_text(
        '{"image_sha256": "a", "classification": "NORMAL"}\n'
        '{"image_sha256": "b", "classification": "DEFECT"}\n',
        encoding="utf-8",
    )
    assert load_completed_hashes(jsonl) == {"a", "b"}


def test_run_batch_skips_existing_hash(tmp_path, mocker):
    from disease_detection.labeling.vlm_client import VLMLabel

    img1 = tmp_path / "a.jpg"
    img1.write_bytes(b"one")
    img2 = tmp_path / "b.jpg"
    img2.write_bytes(b"two")

    jsonl = tmp_path / "out.jsonl"
    # a.jpg is pre-labeled
    jsonl.write_text(
        '{"image_sha256": "' + hash_image_file(img1) + '", "severity": 0}\n',
        encoding="utf-8",
    )

    call_mock = mocker.patch(
        "disease_detection.labeling.batch_label.call_claude_cli",
        return_value=VLMLabel(classification="DEFECT", severity=7, explanation="x"),
    )

    jobs = [
        BatchJob(image_path=img1, crop="pear", plant_part="leaf"),
        BatchJob(image_path=img2, crop="pear", plant_part="leaf"),
    ]
    result: BatchResult = run_batch(
        jobs=jobs,
        output_jsonl=jsonl,
        prompt="P",
        prompt_version="v1",
        model="haiku",
    )

    assert result.processed == 1  # only b.jpg
    assert result.skipped == 1
    assert call_mock.call_count == 1


def test_run_batch_writes_errors(tmp_path, mocker):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"one")
    jsonl = tmp_path / "out.jsonl"

    mocker.patch(
        "disease_detection.labeling.batch_label.call_claude_cli",
        side_effect=RuntimeError("rate limit"),
    )
    # minimize backoff for test
    mocker.patch("disease_detection.labeling.batch_label.time.sleep", return_value=None)

    jobs = [BatchJob(image_path=img, crop="pear", plant_part="leaf")]
    result = run_batch(
        jobs=jobs,
        output_jsonl=jsonl,
        prompt="P",
        prompt_version="v1",
        model="haiku",
        max_retries=2,
    )
    assert result.failed == 1
    errors_path = jsonl.with_name(jsonl.stem + ".errors.jsonl")
    assert errors_path.exists()
    assert "rate limit" in errors_path.read_text()

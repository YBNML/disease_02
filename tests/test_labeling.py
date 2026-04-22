from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from disease_detection.labeling.batch_label import (
    BatchJob,
    BatchResult,
    hash_image_file,
    load_completed_hashes,
    run_batch,
)
from disease_detection.labeling.prompts import PROMPT_VERSION, SEVERITY_PROMPT_V2
from disease_detection.labeling.vlm_client import (
    PartLabel,
    VLMLabel,
    call_claude_cli,
    parse_vlm_response,
)


# ── prompts ──────────────────────────────────────────────────────────────


def test_prompt_version_bumped_to_v2():
    assert PROMPT_VERSION == "v2"


def test_prompt_contains_four_parts():
    for part in ("leaf", "branch", "fruit", "stem"):
        assert part in SEVERITY_PROMPT_V2


def test_prompt_contains_state_and_severity():
    assert "state" in SEVERITY_PROMPT_V2
    assert "severity" in SEVERITY_PROMPT_V2.lower()
    for state in ("defect", "normal", "absent"):
        assert state in SEVERITY_PROMPT_V2


# ── parse_vlm_response (v2) ──────────────────────────────────────────────


_V2_GOOD = json.dumps(
    {
        "parts": {
            "leaf":   {"state": "defect", "severity": 6, "reason": "browning"},
            "branch": {"state": "normal", "severity": 0, "reason": ""},
            "fruit":  {"state": "absent", "severity": 0, "reason": ""},
            "stem":   {"state": "absent", "severity": 0, "reason": ""},
        }
    }
)


def test_parse_vlm_response_v2_parses_all_parts():
    label = parse_vlm_response(_V2_GOOD)
    assert isinstance(label, VLMLabel)
    assert set(label.parts.keys()) == {"leaf", "branch", "fruit", "stem"}
    assert label.parts["leaf"] == PartLabel(state="defect", severity=6, reason="browning")
    assert label.parts["fruit"].state == "absent"


def test_parse_vlm_response_with_code_fences():
    raw = f"```json\n{_V2_GOOD}\n```"
    label = parse_vlm_response(raw)
    assert label.parts["branch"].state == "normal"


def test_parse_vlm_response_forces_severity_zero_for_non_defect():
    """v2 규칙: normal / absent → severity 자동 0."""
    raw = json.dumps(
        {
            "parts": {
                "leaf":   {"state": "normal", "severity": 5, "reason": "?"},
                "branch": {"state": "defect", "severity": 2, "reason": "x"},
                "fruit":  {"state": "absent", "severity": 9, "reason": ""},
                "stem":   {"state": "normal", "severity": 0, "reason": ""},
            }
        }
    )
    label = parse_vlm_response(raw)
    assert label.parts["leaf"].severity == 0
    assert label.parts["branch"].severity == 2  # defect만 유지
    assert label.parts["fruit"].severity == 0


def test_parse_vlm_response_missing_part_raises():
    raw = json.dumps({"parts": {"leaf": {"state": "normal", "severity": 0}}})
    with pytest.raises(ValueError, match="Missing 'branch'"):
        parse_vlm_response(raw)


def test_parse_vlm_response_invalid_state_raises():
    raw = json.dumps(
        {
            "parts": {
                p: {"state": "bogus", "severity": 0} for p in ("leaf", "branch", "fruit", "stem")
            }
        }
    )
    with pytest.raises(ValueError, match="Unknown part state"):
        parse_vlm_response(raw)


def test_parse_vlm_response_handles_nested_reason_with_braces():
    raw = json.dumps(
        {
            "parts": {
                "leaf":   {"state": "defect", "severity": 3, "reason": "text with } brace"},
                "branch": {"state": "absent", "severity": 0, "reason": ""},
                "fruit":  {"state": "absent", "severity": 0, "reason": ""},
                "stem":   {"state": "absent", "severity": 0, "reason": ""},
            }
        }
    )
    label = parse_vlm_response(raw)
    assert "brace" in label.parts["leaf"].reason


# ── call_claude_cli ──────────────────────────────────────────────────────


def test_call_claude_cli_invokes_subprocess(mocker, tmp_path):
    completed = MagicMock(returncode=0, stdout=_V2_GOOD, stderr="")
    run_mock = mocker.patch(
        "disease_detection.labeling.vlm_client.subprocess.run",
        return_value=completed,
    )

    img = tmp_path / "a.jpg"
    img.write_bytes(b"fake")

    label = call_claude_cli(img, prompt="P", model="haiku")
    assert label.parts["leaf"].state == "defect"

    args, _ = run_mock.call_args
    assert args[0][0] == "claude"
    assert "-p" in args[0]
    assert "--model" in args[0]
    assert "haiku" in args[0]


def test_call_claude_cli_nonzero_exit_raises(mocker, tmp_path):
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
    with pytest.raises(FileNotFoundError):
        call_claude_cli(tmp_path / "nope.jpg", prompt="P", model="haiku")


def test_call_claude_cli_timeout_normalized_to_runtimeerror(mocker, tmp_path):
    mocker.patch(
        "disease_detection.labeling.vlm_client.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=1),
    )
    img = tmp_path / "a.jpg"
    img.write_bytes(b"fake")
    with pytest.raises(RuntimeError, match="timed out"):
        call_claude_cli(img, prompt="P", model="haiku", timeout_seconds=1)


# ── batch_label ──────────────────────────────────────────────────────────


def test_hash_image_file_deterministic(tmp_path):
    p = tmp_path / "x.jpg"
    p.write_bytes(b"hello world")
    h1 = hash_image_file(p)
    h2 = hash_image_file(p)
    assert h1 == h2 and len(h1) == 64


def test_load_completed_hashes_reads_existing_jsonl(tmp_path):
    jsonl = tmp_path / "labels.jsonl"
    jsonl.write_text(
        '{"image_sha256": "a"}\n{"image_sha256": "b"}\n', encoding="utf-8"
    )
    assert load_completed_hashes(jsonl) == {"a", "b"}


def test_run_batch_skips_existing_hash(tmp_path, mocker):
    img1 = tmp_path / "a.jpg"
    img1.write_bytes(b"one")
    img2 = tmp_path / "b.jpg"
    img2.write_bytes(b"two")

    jsonl = tmp_path / "out.jsonl"
    jsonl.write_text(
        '{"image_sha256": "' + hash_image_file(img1) + '"}\n', encoding="utf-8"
    )

    v2_label = VLMLabel(
        parts={
            "leaf":   PartLabel("defect", 5, "x"),
            "branch": PartLabel("absent", 0, ""),
            "fruit":  PartLabel("absent", 0, ""),
            "stem":   PartLabel("absent", 0, ""),
        }
    )
    call_mock = mocker.patch(
        "disease_detection.labeling.batch_label.call_claude_cli",
        return_value=v2_label,
    )

    jobs = [
        BatchJob(image_path=img1, crop="pear"),
        BatchJob(image_path=img2, crop="pear"),
    ]
    result: BatchResult = run_batch(
        jobs=jobs,
        output_jsonl=jsonl,
        prompt="P",
        prompt_version="v2",
        model="haiku",
    )
    assert result.processed == 1
    assert result.skipped == 1
    assert call_mock.call_count == 1

    # 새로 추가된 줄이 v2 스키마인지 확인
    lines = jsonl.read_text(encoding="utf-8").strip().split("\n")
    last = json.loads(lines[-1])
    assert "parts" in last
    assert set(last["parts"].keys()) == {"leaf", "branch", "fruit", "stem"}
    assert last["prompt_version"] == "v2"


def test_run_batch_writes_errors(tmp_path, mocker):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"one")
    jsonl = tmp_path / "out.jsonl"

    mocker.patch(
        "disease_detection.labeling.batch_label.call_claude_cli",
        side_effect=RuntimeError("rate limit"),
    )
    mocker.patch("disease_detection.labeling.batch_label.time.sleep", return_value=None)

    jobs = [BatchJob(image_path=img, crop="pear")]
    result = run_batch(
        jobs=jobs,
        output_jsonl=jsonl,
        prompt="P",
        prompt_version="v2",
        model="haiku",
        max_retries=2,
    )
    assert result.failed == 1
    errors_path = jsonl.with_name(jsonl.stem + ".errors.jsonl")
    assert errors_path.exists()
    assert "rate limit" in errors_path.read_text()

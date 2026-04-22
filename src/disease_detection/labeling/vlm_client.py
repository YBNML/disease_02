"""Claude Code CLI headless 래퍼 + VLM 응답 파서 (prompt v2).

- `call_claude_cli(image_path, prompt, model)`: `claude -p --model <m>` subprocess 호출.
- `parse_vlm_response(raw) -> VLMLabel`: v2 JSON 응답을 구조화해 반환.

v2 응답 포맷 (prompts.SEVERITY_PROMPT_V2):
    {
      "parts": {
        "leaf":   {"state": "defect|normal|absent", "severity": 0-10, "reason": "..."},
        "branch": {...},
        "fruit":  {...},
        "stem":   {...}
      }
    }

재시도·백오프 로직은 `batch_label.py` 가 담당; 이 모듈은 단일 호출·파싱만.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

_VALID_STATES = frozenset({"defect", "normal", "absent"})
_REQUIRED_PARTS = ("leaf", "branch", "fruit", "stem")


@dataclass(frozen=True)
class PartLabel:
    state: str       # "defect" | "normal" | "absent"
    severity: int    # 0–10
    reason: str


@dataclass(frozen=True)
class VLMLabel:
    """`parts` 는 leaf/branch/fruit/stem 4 부위 모두 포함."""

    parts: Mapping[str, PartLabel]


def _extract_first_json_object(raw: str) -> str:
    """첫 balanced `{ ... }` 블록 추출. 문자열 내 중괄호·escape 처리."""
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"JSON 블록을 찾지 못함: {raw[:200]}")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    raise ValueError(f"균형 맞지 않는 JSON 블록: {raw[:200]}")


def _parse_part(raw: dict) -> PartLabel:
    state = str(raw.get("state", "")).lower()
    if state not in _VALID_STATES:
        raise ValueError(f"Unknown part state: {state!r}")
    sev = int(raw.get("severity", 0))
    if not 0 <= sev <= 10:
        raise ValueError(f"Severity out of range: {sev}")
    if state in {"normal", "absent"} and sev != 0:
        # v2 규칙: defect 만 1~10. 다른 상태는 0 강제.
        sev = 0
    reason = str(raw.get("reason", "")).strip()
    return PartLabel(state=state, severity=sev, reason=reason)


def parse_vlm_response(raw: str) -> VLMLabel:
    """v2 JSON 응답을 `VLMLabel` 로 변환."""
    obj = json.loads(_extract_first_json_object(raw))
    parts_raw = obj.get("parts")
    if not isinstance(parts_raw, dict):
        raise ValueError("Missing or invalid 'parts' object in response")
    parts: dict[str, PartLabel] = {}
    for name in _REQUIRED_PARTS:
        if name not in parts_raw:
            raise ValueError(f"Missing '{name}' part in response")
        parts[name] = _parse_part(parts_raw[name])
    return VLMLabel(parts=parts)


def call_claude_cli(
    image_path: Path,
    prompt: str,
    model: str = "haiku",
    timeout_seconds: int = 120,
) -> VLMLabel:
    """`claude -p` 호출 후 v2 응답 파싱.

    이미지 경로는 프롬프트 끝에 포함. Claude Code는 절대 경로를 Read 툴로 자동 로딩.

    Raises:
        FileNotFoundError: `image_path` 없음.
        RuntimeError: CLI 비정상 종료, timeout, 또는 응답 파싱 실패.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"이미지 없음: {image_path}")
    full_prompt = f"{prompt}\n\nImage to inspect: {image_path}"
    args = ["claude", "-p", full_prompt, "--model", model]
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"claude CLI timed out after {timeout_seconds}s: {exc}"
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    return parse_vlm_response(result.stdout)

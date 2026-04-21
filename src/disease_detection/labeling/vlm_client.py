"""Claude Code CLI headless 래퍼.

`claude -p "<prompt>" --model <model>` 를 subprocess로 호출하고 JSON 응답 파싱.
실패 시 재시도 로직은 `batch_label.py`에서 담당; 이 모듈은 단일 호출·파싱만.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VLMLabel:
    classification: str  # NORMAL or DEFECT
    severity: int  # 0-10
    explanation: str


_JSON_BLOCK = re.compile(r"\{.*?\}", re.DOTALL)


def parse_vlm_response(raw: str) -> VLMLabel:
    """문자열에서 첫 번째 JSON 블록을 추출하여 VLMLabel로 변환."""
    match = _JSON_BLOCK.search(raw)
    if match is None:
        raise ValueError(f"JSON 블록을 찾지 못함: {raw[:200]}")
    obj = json.loads(match.group(0))
    cls = str(obj["classification"]).upper()
    sev = int(obj["severity"])
    expl = str(obj.get("explanation", ""))
    if cls not in {"NORMAL", "DEFECT"}:
        raise ValueError(f"Unknown classification: {cls}")
    if not 0 <= sev <= 10:
        raise ValueError(f"Severity out of range: {sev}")
    return VLMLabel(classification=cls, severity=sev, explanation=expl)


def call_claude_cli(
    image_path: Path,
    prompt: str,
    model: str = "haiku",
    timeout_seconds: int = 120,
) -> VLMLabel:
    """`claude -p` 호출 후 응답 파싱.

    이미지 경로는 프롬프트 끝에 직접 포함. Claude Code는 절대 경로를 Read 툴로 자동 로딩.
    """
    full_prompt = f"{prompt}\n\nImage to inspect: {image_path}"
    args = ["claude", "-p", full_prompt, "--model", model]
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    return parse_vlm_response(result.stdout)

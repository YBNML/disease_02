"""재개 가능한 VLM 배치 라벨링 실행기.

- SHA256 기반 중복 회피 (이미 JSONL에 있는 해시는 스킵)
- 이미지당 exponential backoff 재시도, 영구 실패는 `.errors.jsonl`에 보존
- 진행 상황은 tqdm으로 표시
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from .vlm_client import VLMLabel, call_claude_cli


@dataclass(frozen=True)
class BatchJob:
    """범용 결함 라벨링 배치 단위 — 이미지 1장 (bbox 1개) 에 대응.

    AIhub 는 이미지당 bbox 1개이고 VLM v2 는 이미지 crop을 통째로 본 뒤 4부위를
    동시에 판정하므로 별도 `plant_part` 필드가 필요 없다.
    """

    image_path: Path
    crop: str


@dataclass
class BatchResult:
    processed: int
    skipped: int
    failed: int


def hash_image_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_completed_hashes(jsonl: Path) -> set[str]:
    if not jsonl.exists():
        return set()
    done: set[str] = set()
    with jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "image_sha256" in obj:
                    done.add(obj["image_sha256"])
            except json.JSONDecodeError:
                continue
    return done


def _backoff_seconds(attempt: int, base: int = 60, cap: int = 1800) -> int:
    return min(cap, base * (2 ** (attempt - 1)))


def _write_label_line(
    output: Path,
    *,
    job: BatchJob,
    sha: str,
    label: VLMLabel,
    model: str,
    prompt_version: str,
) -> None:
    parts_record = {
        name: {
            "state": p.state,
            "severity": p.severity,
            "reason": p.reason,
        }
        for name, p in label.parts.items()
    }
    record = {
        "image_path": str(job.image_path),
        "image_sha256": sha,
        "crop": job.crop,
        "parts": parts_record,
        "model": model,
        "prompt_version": prompt_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with output.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_error_line(errors: Path, *, job: BatchJob, error: str) -> None:
    errors.parent.mkdir(parents=True, exist_ok=True)
    with errors.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "image_path": str(job.image_path),
                    "crop": job.crop,
                    "error": error,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            + "\n"
        )


def run_batch(
    jobs: list[BatchJob],
    output_jsonl: Path,
    prompt: str,
    prompt_version: str,
    model: str = "haiku",
    max_retries: int = 3,
) -> BatchResult:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    errors_path = output_jsonl.with_name(output_jsonl.stem + ".errors.jsonl")
    done_hashes = load_completed_hashes(output_jsonl)

    processed = skipped = failed = 0
    for job in tqdm(jobs, desc="VLM labeling"):
        sha = hash_image_file(job.image_path)
        if sha in done_hashes:
            skipped += 1
            continue
        last_error: str | None = None
        for attempt in range(1, max_retries + 1):
            try:
                label = call_claude_cli(job.image_path, prompt=prompt, model=model)
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt < max_retries:
                    time.sleep(_backoff_seconds(attempt))
                continue
            _write_label_line(
                output_jsonl,
                job=job,
                sha=sha,
                label=label,
                model=model,
                prompt_version=prompt_version,
            )
            done_hashes.add(sha)
            processed += 1
            break
        else:
            _write_error_line(errors_path, job=job, error=last_error or "unknown")
            failed += 1
    return BatchResult(processed=processed, skipped=skipped, failed=failed)

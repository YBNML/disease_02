"""경로 해석 유틸. DATASET_ROOT 환경변수를 단일 진입점으로 관리."""
from __future__ import annotations

import os
from pathlib import Path


def resolve_dataset_root(require_exists: bool = True) -> Path:
    """DATASET_ROOT 환경변수에서 데이터셋 루트 경로를 반환.

    Args:
        require_exists: True면 경로 존재 검증. False면 경로 형식만 반환.

    Raises:
        RuntimeError: DATASET_ROOT 미설정.
        FileNotFoundError: require_exists=True인데 경로 없음.
    """
    raw = os.environ.get("DATASET_ROOT")
    if not raw:
        raise RuntimeError(
            "DATASET_ROOT 환경변수가 설정되지 않음. .env.example 참고."
        )
    path = Path(raw).expanduser()
    if require_exists and not path.exists():
        raise FileNotFoundError(f"DATASET_ROOT 경로 없음: {path}")
    return path

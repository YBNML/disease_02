#!/usr/bin/env python3
"""AIhub 「과수화상병 촬영 이미지」 zip 을 우리 레이아웃으로 추출·재배치.

AIhub 다운로드 구조 (입력):
    <SRC>/
      Training/
        [라벨]배_0.정상.zip, [라벨]배_1.질병.zip,
        [라벨]사과_0.정상.zip, [라벨]사과_1.질병.zip,
        [원천]배_0.정상_(1).zip, [원천]배_0.정상_(2).zip, [원천]배_1.질병.zip,
        [원천]사과_0.정상_(1).zip, [원천]사과_0.정상_(2).zip, [원천]사과_1.질병.zip
      Validation/
        [라벨]배_0.정상.zip, [라벨]배_1.질병.zip, [라벨]사과_0.정상.zip, [라벨]사과_1.질병.zip
        [원천]배_0.정상.zip, [원천]배_1.질병.zip, [원천]사과_0.정상.zip, [원천]사과_1.질병.zip

타겟 레이아웃 (`load_aihub_split` 기대):
    <DATASET_ROOT>/raw/aihub/
      pear/
        images/*.{jpg,JPG,png}
        annotations/*.jpg.json
      apple/
        images/...
        annotations/...

특징:
- Training / Validation 구분은 버림 (자체 stratified split 사용).
- 정상 / 질병 구분도 파일 레벨에서 버림 (라벨 JSON 의 `disease` 필드로 구별).
- `.irx577` 다운로드 fragment 는 무시.
- 이미 추출된 zip 은 `.extracted` sentinel 로 스킵 (재개 지원).
- 라벨 zip 만 먼저 처리하려면 `--labels-only`.

실행 예:
    python scripts/extract_aihub.py \
        --src "/Users/khj/YBNML_macmini/disease_02/과수화상병 촬영 이미지"
    # → $DATASET_ROOT/raw/aihub/{pear,apple}/{images,annotations} 생성
"""
from __future__ import annotations

import argparse
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from disease_detection.utils.io import resolve_dataset_root

# 한글 키워드 매핑.
_CROP_TOKEN = {"배": "pear", "사과": "apple"}
# `[라벨]` / `[원천]` 판별.
_LABEL_TAG = "[라벨]"
_SOURCE_TAG = "[원천]"


@dataclass(frozen=True)
class ZipJob:
    zip_path: Path
    crop: str            # "pear" | "apple"
    kind: str            # "annotations" | "images"


_FILENAME_RE = re.compile(r"\[(라벨|원천)\](배|사과)_[01]\.(정상|질병)(?:_\(\d+\))?\.zip$")


def parse_zip_filename(p: Path) -> ZipJob | None:
    """`[라벨]배_0.정상.zip` 같은 이름을 `ZipJob` 으로 파싱. 매칭 안 되면 None."""
    m = _FILENAME_RE.match(p.name)
    if m is None:
        return None
    tag, crop_hangul, _ = m.group(1), m.group(2), m.group(3)
    crop = _CROP_TOKEN[crop_hangul]
    kind = "annotations" if tag == "라벨" else "images"
    return ZipJob(zip_path=p, crop=crop, kind=kind)


def collect_jobs(src: Path, labels_only: bool = False) -> list[ZipJob]:
    """`src/Training/*.zip` + `src/Validation/*.zip` 에서 ZipJob 수집."""
    jobs: list[ZipJob] = []
    for split_dir_name in ("Training", "Validation"):
        split_dir = src / split_dir_name
        if not split_dir.exists():
            continue
        for zip_path in sorted(split_dir.glob("*.zip")):
            job = parse_zip_filename(zip_path)
            if job is None:
                continue
            if labels_only and job.kind != "annotations":
                continue
            jobs.append(job)
    return jobs


def _sentinel_path(target_dir: Path, zip_path: Path) -> Path:
    """Sentinel 은 target 디렉토리 안의 hidden 파일로 둔다 — target 별로 독립 추적."""
    return target_dir / f".{zip_path.name}.extracted"


def extract_one(job: ZipJob, target_root: Path, force: bool = False) -> int:
    """단일 zip 추출. 이미 추출된 경우 sentinel 로 스킵. 추출된 파일 수 반환."""
    target_dir = target_root / "raw" / "aihub" / job.crop / job.kind
    target_dir.mkdir(parents=True, exist_ok=True)
    sentinel = _sentinel_path(target_dir, job.zip_path)
    if sentinel.exists() and not force:
        return 0

    with zipfile.ZipFile(job.zip_path, "r") as zf:
        names = zf.namelist()
        # 중첩 디렉토리 무시 — 우리는 flat 구조를 원함.
        count = 0
        for name in tqdm(names, desc=job.zip_path.name, leave=False):
            if name.endswith("/"):
                continue
            # 파일명만 추출 (경로 prefix 제거).
            flat_name = Path(name).name
            out = target_dir / flat_name
            if out.exists() and not force:
                continue
            with zf.open(name) as src_fh, out.open("wb") as dst_fh:
                dst_fh.write(src_fh.read())
            count += 1
    sentinel.write_text(f"extracted {count} files\n", encoding="utf-8")
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src", type=Path, required=True,
        help="AIhub 데이터 디렉토리 (Training/ Validation/ 포함)",
    )
    parser.add_argument(
        "--labels-only", action="store_true",
        help="`[라벨]` zip 만 추출 (빠른 스키마 점검용)",
    )
    parser.add_argument("--force", action="store_true", help="이미 추출된 것도 재추출")
    args = parser.parse_args()

    src: Path = args.src
    if not src.exists():
        print(f"[err] source 디렉토리 없음: {src}")
        return 2

    target_root = resolve_dataset_root(require_exists=False)
    target_root.mkdir(parents=True, exist_ok=True)

    jobs = collect_jobs(src, labels_only=args.labels_only)
    if not jobs:
        print(f"[err] 추출할 zip 이 없습니다 (src={src}, labels_only={args.labels_only})")
        return 1

    print(f"[info] {len(jobs)} zip 추출 예정 → {target_root}/raw/aihub/")
    total = 0
    for job in jobs:
        n = extract_one(job, target_root, force=args.force)
        total += n
        print(f"  ✓ {job.zip_path.name} → {job.crop}/{job.kind}/ ({n} files)")
    print(f"[done] 총 {total} 파일 추출됨 (sentinel 로 resume-가능).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

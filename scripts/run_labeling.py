#!/usr/bin/env python3
"""범용 결함 분류기용 VLM 재라벨링 실행기 (macmini).

샘플링: crop × plant_part × fireblight 균등 분포에서 총 N 장 선택.
실행 예:
    python scripts/run_labeling.py --sample-size 3000 --out labels/vlm_severity/severity_labels.jsonl
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from disease_detection.data.aihub import load_aihub_split
from disease_detection.labeling.batch_label import BatchJob, run_batch
from disease_detection.labeling.prompts import PROMPT_VERSION, SEVERITY_PROMPT_V1
from disease_detection.utils.io import resolve_dataset_root
from disease_detection.utils.seeding import set_seed


def collect_jobs(aihub_root: Path, sample_size: int, seed: int) -> list[BatchJob]:
    """crop × plant_part × fireblight 균등 stratified 샘플링."""
    buckets: dict[tuple[str, str, int], list[BatchJob]] = defaultdict(list)
    for crop_name in ("pear", "apple"):
        root = aihub_root / crop_name
        if not root.exists():
            continue
        for entry in load_aihub_split(root):
            for box in entry.boxes:
                key = (entry.crop, box.category, entry.fireblight)
                buckets[key].append(
                    BatchJob(
                        image_path=entry.image_path,
                        crop=entry.crop,
                        plant_part=box.category,
                    )
                )

    rng = random.Random(seed)
    if not buckets:
        return []
    # sample_size < len(buckets)인 경우: 버킷을 결정적으로 sub-sample하여 1개씩 뽑음.
    # 이렇게 하지 않으면 per_bucket이 1로 floor되어 모든 버킷에서 뽑다가 끝에서
    # 잘려나가, 남는 버킷이 dict 삽입 순서에 의존해 비결정적 strata selection이 됨.
    bucket_keys = list(buckets.keys())
    if sample_size < len(bucket_keys):
        bucket_keys = rng.sample(bucket_keys, sample_size)
        per_bucket = 1
    else:
        per_bucket = max(1, sample_size // len(bucket_keys))
    selected: list[BatchJob] = []
    for key in bucket_keys:
        jobs = buckets[key]
        rng.shuffle(jobs)
        selected.extend(jobs[:per_bucket])
    rng.shuffle(selected)
    return selected[:sample_size]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("labels/vlm_severity/severity_labels.jsonl"),
    )
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)
    ds_root = resolve_dataset_root()
    aihub_root = ds_root / "raw" / "aihub"

    jobs = collect_jobs(aihub_root, args.sample_size, args.seed)
    out_path = ds_root / args.out
    result = run_batch(
        jobs=jobs,
        output_jsonl=out_path,
        prompt=SEVERITY_PROMPT_V1,
        prompt_version=PROMPT_VERSION,
        model=args.model,
        max_retries=args.max_retries,
    )
    print(
        f"processed={result.processed} skipped={result.skipped} failed={result.failed}"
    )
    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

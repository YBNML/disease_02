#!/usr/bin/env python3
"""학습용 split 생성기 (Ubuntu).

- detector_split.json — detection용 train/val/test 이미지 인덱스.
- classifier_fireblight_split.json — 이미지 단위 train/val/test, AIhub 원본 라벨.
- classifier_defect_split.json — VLM 라벨링된 subset의 crop 단위 split.
결정성: 고정 seed.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from disease_detection.data.aihub import load_aihub_split
from disease_detection.data.classification_dataset import (
    build_defect_crops,
    build_fireblight_crops,
)
from disease_detection.utils.io import resolve_dataset_root


def stratified_split(
    items: list,
    keys: list,
    ratios=(0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, list[int]]:
    assert len(items) == len(keys)
    rng = random.Random(seed)
    buckets: dict = defaultdict(list)
    for idx, key in enumerate(keys):
        buckets[key].append(idx)
    train, val, test = [], [], []
    for key, idxs in buckets.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        train.extend(idxs[:n_train])
        val.extend(idxs[n_train : n_train + n_val])
        test.extend(idxs[n_train + n_val :])
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("splits"),
    )
    parser.add_argument(
        "--vlm-jsonl",
        type=Path,
        default=Path("labels/vlm_severity/severity_labels.jsonl"),
    )
    parser.add_argument("--defect-threshold", type=int, default=4)
    args = parser.parse_args()

    ds_root = resolve_dataset_root()
    aihub_root = ds_root / "raw" / "aihub"
    out_dir = ds_root / args.splits_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    det_entries = []
    for crop_name in ("pear", "apple"):
        det_entries.extend(load_aihub_split(aihub_root / crop_name))
    det_keys = [f"{e.crop}-leaf_count:{sum(1 for b in e.boxes if b.category == 'leaf')}" for e in det_entries]
    det_split = stratified_split(det_entries, det_keys, seed=args.seed)
    (out_dir / "detector_split.json").write_text(
        json.dumps(det_split, indent=2), encoding="utf-8"
    )

    # 분류기 split — 이미지 단위로 먼저 split 하고 crop 배분
    fb_items = []
    for crop_name in ("pear", "apple"):
        fb_items.extend(build_fireblight_crops(aihub_root / crop_name))
    fb_keys = [f"{it.crop}-{it.label}" for it in fb_items]
    fb_split = stratified_split(fb_items, fb_keys, seed=args.seed)
    (out_dir / "classifier_fireblight_split.json").write_text(
        json.dumps(fb_split, indent=2), encoding="utf-8"
    )

    vlm_path = ds_root / args.vlm_jsonl
    if vlm_path.exists():
        def_items = []
        for crop_name in ("pear", "apple"):
            def_items.extend(
                build_defect_crops(
                    aihub_root / crop_name,
                    vlm_path,
                    defect_threshold=args.defect_threshold,
                )
            )
        def_keys = [f"{it.crop}-{it.plant_part}-{it.label}" for it in def_items]
        def_split = stratified_split(def_items, def_keys, seed=args.seed)
        (out_dir / "classifier_defect_split.json").write_text(
            json.dumps(def_split, indent=2), encoding="utf-8"
        )
        print(f"defect items: {len(def_items)}")

    print(
        f"detector items: {len(det_entries)}  fireblight items: {len(fb_items)}  "
        f"output: {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

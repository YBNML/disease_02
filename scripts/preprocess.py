#!/usr/bin/env python3
"""학습용 split 생성기 (Ubuntu).

- `detector_split.json` — Faster R-CNN single-class plant_roi 학습용 이미지 split.
- `classifier_fireblight_split.json` — ResNet18 binary 학습용 (AIhub 이진화 라벨).
- `classifier_defect_split.json` — MultiPartDefectClassifier 학습용 (VLM v2 라벨이 있는
  subset 만).
결정성: 고정 seed stratified.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from disease_detection.data.aihub import load_aihub_split
from disease_detection.data.classification_dataset import (
    PLANT_PARTS,
    build_defect_items,
    build_fireblight_items,
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
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument(
        "--vlm-jsonl",
        type=Path,
        default=Path("labels/vlm_severity/severity_labels.jsonl"),
    )
    args = parser.parse_args()

    ds_root = resolve_dataset_root()
    aihub_root = ds_root / "raw" / "aihub"
    out_dir = ds_root / args.splits_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Detector split (single-class plant_roi) — image-level, stratified by
    #    (crop, fireblight) 로 정상·질병 비율 균형.
    det_entries = []
    for crop_name in ("pear", "apple"):
        det_entries.extend(load_aihub_split(aihub_root / crop_name))
    det_keys = [f"{e.crop}-{e.fireblight}" for e in det_entries]
    det_split = stratified_split(det_entries, det_keys, seed=args.seed)
    (out_dir / "detector_split.json").write_text(
        json.dumps(det_split, indent=2), encoding="utf-8"
    )

    # ── Fireblight classifier split (binary)
    fb_items = []
    for crop_name in ("pear", "apple"):
        fb_items.extend(build_fireblight_items(aihub_root / crop_name))
    fb_keys = [f"{it.crop}-{it.label}" for it in fb_items]
    fb_split = stratified_split(fb_items, fb_keys, seed=args.seed)
    (out_dir / "classifier_fireblight_split.json").write_text(
        json.dumps(fb_split, indent=2), encoding="utf-8"
    )

    # ── Defect classifier split (multi-part, VLM 라벨 존재하는 것만)
    vlm_path = ds_root / args.vlm_jsonl
    n_def = 0
    if vlm_path.exists():
        def_items = []
        for crop_name in ("pear", "apple"):
            def_items.extend(
                build_defect_items(aihub_root / crop_name, vlm_path)
            )
        # stratification key: crop + 각 부위 state 를 연결 (고차원 but deterministic).
        def_keys = [
            f"{it.crop}-" + "-".join(str(s) for s in it.part_states)
            for it in def_items
        ]
        def_split = stratified_split(def_items, def_keys, seed=args.seed)
        (out_dir / "classifier_defect_split.json").write_text(
            json.dumps(def_split, indent=2), encoding="utf-8"
        )
        n_def = len(def_items)

    print(
        f"detector items: {len(det_entries)}  fireblight items: {len(fb_items)}  "
        f"defect items: {n_def}  output: {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

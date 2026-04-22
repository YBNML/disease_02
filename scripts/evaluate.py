#!/usr/bin/env python3
"""Detector + 두 분류기 end-to-end 평가 (Oracle + Realistic).

train split 오염을 피하기 위해 `splits/*.json` 의 test 인덱스만 사용한다.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from disease_detection.data.aihub import load_aihub_split
from disease_detection.data.classification_dataset import (
    build_defect_items,
    build_fireblight_items,
)
from disease_detection.eval.inference import (
    evaluate_classifier_oracle,
    evaluate_multipart_oracle,
    evaluate_pipeline_realistic_image,
)
from disease_detection.models.classifier import PlantDefectClassifier
from disease_detection.models.detector import FasterRCNNModule
from disease_detection.models.multipart_classifier import MultiPartDefectClassifier
from disease_detection.models.pipeline import TwoStagePipeline


def _load_detector(ckpt: Path, device: str) -> FasterRCNNModule:
    m = FasterRCNNModule.load_from_checkpoint(str(ckpt), map_location=device)
    m.eval()
    return m


def _load_fireblight_classifier(ckpt: Path, device: str) -> PlantDefectClassifier:
    m = PlantDefectClassifier.load_from_checkpoint(str(ckpt), map_location=device)
    m.eval()
    return m


def _load_multipart_classifier(ckpt: Path, device: str) -> MultiPartDefectClassifier:
    m = MultiPartDefectClassifier.load_from_checkpoint(str(ckpt), map_location=device)
    m.eval()
    return m


def _test_slice(items: list, split_path: Path) -> list:
    """split JSON 의 test 인덱스로 items 를 슬라이스. 누락 시 전체 반환 + 경고."""
    if not split_path.exists():
        print(
            f"[warn] split file missing: {split_path} — using all items "
            f"(train-contaminated)"
        )
        return items
    split = json.loads(split_path.read_text(encoding="utf-8"))
    return [items[i] for i in split["test"]]


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_root = Path(cfg.paths.dataset_root)
    aihub_root = ds_root / "raw" / "aihub"
    splits_dir = ds_root / "splits"

    fb_ckpt = Path(cfg["fireblight_ckpt"])
    def_ckpt = Path(cfg["defect_ckpt"])
    det_ckpt = Path(cfg["detector_ckpt"])

    det_module = _load_detector(det_ckpt, device)
    fb_module = _load_fireblight_classifier(fb_ckpt, device)
    def_module = _load_multipart_classifier(def_ckpt, device)

    # ── Oracle A: fireblight binary (test split 한정)
    fb_items_all: list = []
    for crop_name in ("pear", "apple"):
        fb_items_all.extend(build_fireblight_items(aihub_root / crop_name))
    fb_test = _test_slice(
        fb_items_all, splits_dir / "classifier_fireblight_split.json"
    )
    fb_report = evaluate_classifier_oracle(
        fb_test,
        classifier_fn=lambda x: fb_module(x.to(device)).detach().cpu(),
    )

    # ── Oracle B: 범용 결함 multi-part (VLM 라벨 있을 때만, test split 한정)
    def_report = None
    n_def = 0
    vlm_path = ds_root / "labels" / "vlm_severity" / "severity_labels.jsonl"
    if vlm_path.exists():
        def_items_all: list = []
        for crop_name in ("pear", "apple"):
            def_items_all.extend(
                build_defect_items(aihub_root / crop_name, vlm_path)
            )
        def_test = _test_slice(
            def_items_all, splits_dir / "classifier_defect_split.json"
        )
        n_def = len(def_test)
        def_report = evaluate_multipart_oracle(
            def_test,
            classifier_fn=lambda x: def_module(x.to(device)).detach().cpu(),
        )

    # ── Realistic: 파이프라인 예측 기반 이미지 단위 화상병 (test split 한정)
    pipeline = TwoStagePipeline(
        detector=lambda imgs: det_module([i.to(device) for i in imgs]),
        fireblight_classifier=lambda c: fb_module(c.to(device)).detach().cpu(),
        defect_classifier=lambda c: def_module(c.to(device)).detach().cpu(),
    )
    all_entries: list = []
    for crop_name in ("pear", "apple"):
        all_entries.extend(load_aihub_split(aihub_root / crop_name))
    det_split_path = splits_dir / "detector_split.json"
    test_entries = _test_slice(all_entries, det_split_path)
    fb_realistic = evaluate_pipeline_realistic_image(test_entries, pipeline)

    out_root = Path("reports") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root.mkdir(parents=True, exist_ok=True)
    report = {
        "split": "test",
        "n_fireblight_crops": len(fb_test),
        "n_defect_crops": n_def,
        "n_realistic_images": len(test_entries),
        "fireblight_oracle": fb_report.__dict__,
        "fireblight_realistic_image_level": fb_realistic.__dict__,
        "defect_oracle_multipart": def_report if def_report else None,
        "ckpts": {
            "detector": str(det_ckpt),
            "fireblight": str(fb_ckpt),
            "defect": str(def_ckpt),
        },
    }
    (out_root / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"report saved: {out_root/'metrics.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Detector + 두 분류기 end-to-end 평가 (Oracle + Realistic)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from disease_detection.data.classification_dataset import (
    build_defect_crops,
    build_fireblight_crops,
)
from disease_detection.eval.inference import evaluate_classifier_oracle
from disease_detection.models.classifier import PlantDefectClassifier
from disease_detection.models.detector import FasterRCNNModule
from disease_detection.models.pipeline import TwoStagePipeline


def _load_classifier(ckpt: Path, device: str) -> PlantDefectClassifier:
    m = PlantDefectClassifier.load_from_checkpoint(str(ckpt), map_location=device)
    m.eval()
    return m


def _load_detector(ckpt: Path, device: str) -> FasterRCNNModule:
    m = FasterRCNNModule.load_from_checkpoint(str(ckpt), map_location=device)
    m.eval()
    return m


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_root = Path(cfg.paths.dataset_root)
    aihub_root = ds_root / "raw" / "aihub"

    fb_ckpt = Path(cfg["fireblight_ckpt"])
    def_ckpt = Path(cfg["defect_ckpt"])
    det_ckpt = Path(cfg["detector_ckpt"])

    det_module = _load_detector(det_ckpt, device)
    fb_module = _load_classifier(fb_ckpt, device)
    def_module = _load_classifier(def_ckpt, device)

    # Oracle evaluation: GT bbox로 crop → 각 분류기 성능
    fb_items = []
    for crop_name in ("pear", "apple"):
        fb_items.extend(build_fireblight_crops(aihub_root / crop_name))
    fb_report = evaluate_classifier_oracle(
        fb_items, classifier_fn=lambda x: fb_module(x.to(device)).detach().cpu()
    )

    def_items = []
    vlm_path = ds_root / "labels" / "vlm_severity" / "severity_labels.jsonl"
    if vlm_path.exists():
        for crop_name in ("pear", "apple"):
            def_items.extend(build_defect_crops(aihub_root / crop_name, vlm_path))
        def_report = evaluate_classifier_oracle(
            def_items,
            classifier_fn=lambda x: def_module(x.to(device)).detach().cpu(),
        )
    else:
        def_report = None

    # Realistic: 파이프라인으로 이미지 단위 예측 → 이미지 단위 GT와 비교 (화상병만)
    pipeline = TwoStagePipeline(
        detector=lambda imgs: det_module([i.to(device) for i in imgs]),
        fireblight_classifier=lambda c: fb_module(c.to(device)).detach().cpu(),
        defect_classifier=lambda c: def_module(c.to(device)).detach().cpu(),
    )

    from disease_detection.data.aihub import load_aihub_split
    from disease_detection.eval.metrics import compute_classification_report

    fb_image_scores: list[float] = []
    fb_image_labels: list[int] = []
    for crop_name in ("pear", "apple"):
        for entry in load_aihub_split(aihub_root / crop_name):
            pred = pipeline.predict_image(entry.image_path, crop=entry.crop)
            score = (
                max((d.fireblight_prob for d in pred.detections), default=0.0)
            )
            fb_image_scores.append(score)
            fb_image_labels.append(int(entry.fireblight))
    fb_realistic = compute_classification_report(
        torch.tensor(fb_image_labels), torch.tensor(fb_image_scores)
    )

    out_root = Path("reports") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root.mkdir(parents=True, exist_ok=True)
    report = {
        "fireblight_oracle": fb_report.__dict__,
        "fireblight_realistic_image_level": fb_realistic.__dict__,
        "defect_oracle": def_report.__dict__ if def_report else None,
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

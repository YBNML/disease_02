# Phase 1 구현 계획 — 배·사과 병충해 검출

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 원본 PPT의 D → A 파이프라인(VLM 심각도 라벨링 + Faster R-CNN + ResNet18 × 2 독립 분류기)을 PyTorch Lightning + Hydra 기반으로 재구현하여 배·사과 화상병·범용 결함 검출 baseline을 확보한다.

**Architecture:** macmini에서 AIhub 데이터 수집 → Claude Code CLI(Haiku)로 범용 결함 심각도 라벨링 → Ubuntu로 전송 → Fast R-CNN이 부위 bbox 예측 → GT/예측 bbox로 crop → 독립 ResNet18 두 개가 화상병/범용 결함 이진 판정. 평가는 Oracle(GT bbox) / Realistic(예측 bbox) 양쪽 측정.

**Tech Stack:** Python 3.11, PyTorch, PyTorch Lightning, torchvision (v2 transforms, FasterRCNN, ResNet18), torchmetrics, Hydra, pytest, conda, Claude Code CLI (headless Haiku).

**참고 문서:** 프로젝트 루트 `README.md` (Phase 1 설계 전체)

---

## 파일 구조 (최종 상태)

```
src/disease_detection/
  __init__.py
  utils/{__init__.py, io.py, seeding.py}
  data/{__init__.py, aihub.py, detection_dataset.py, classification_dataset.py, transforms.py}
  labeling/{__init__.py, prompts.py, vlm_client.py, batch_label.py}
  models/{__init__.py, detector.py, classifier.py, pipeline.py}
  eval/{__init__.py, metrics.py, inference.py}

configs/
  config.yaml
  data/{aihub_pear.yaml, aihub_apple.yaml, aihub_combined.yaml}
  model/{detector_fasterrcnn.yaml, classifier_resnet18.yaml}
  trainer/{local_cpu.yaml, ubuntu_gpu.yaml}
  experiment/{fireblight_baseline.yaml, defect_baseline.yaml}

scripts/
  download_aihub.sh
  run_labeling.py
  sync_to_ubuntu.sh
  preprocess.py
  train_detector.py
  train_classifier.py
  evaluate.py

tests/
  conftest.py
  fixtures/{dummy_aihub/, dummy_vlm_labels.jsonl}
  test_utils.py
  test_data.py
  test_labeling.py
  test_models.py
  test_pipeline.py
  test_eval.py

pyproject.toml
environment-common.yml
environment-macos.yml
environment-linux.yml
.env.example
README.md  (존재, 구현 완료 후 사용법 섹션 추가)
```

**책임 분리 원칙:** 각 파일은 단일 관심사만 담당. 예: `aihub.py`는 AIhub 원본 어노테이션을 공통 dataclass로 파싱만. 데이터셋 클래스는 파싱 결과를 torch Dataset으로 감싸기만. 모델 코드는 Lightning 로직만.

---

## Task 1: 프로젝트 스켈레톤 & conda 환경

**Files:**
- Create: `pyproject.toml`
- Create: `environment-common.yml`
- Create: `environment-macos.yml`
- Create: `environment-linux.yml`
- Create: `.env.example`
- Create: `src/disease_detection/__init__.py`
- Create: `src/disease_detection/{utils,data,labeling,models,eval}/__init__.py` (5개)
- Create: `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: `pyproject.toml` 작성**

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "disease-detection"
version = "0.1.0"
description = "Pear & apple disease detection — Phase 1 baseline"
requires-python = ">=3.11"
dependencies = [
  "torch>=2.3",
  "torchvision>=0.18",
  "pytorch-lightning>=2.3",
  "torchmetrics>=1.4",
  "hydra-core>=1.3",
  "omegaconf>=2.3",
  "pyyaml",
  "pillow",
  "numpy",
  "tqdm",
  "python-dotenv",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-mock", "ruff", "black"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "integration: requires real Claude CLI (manual only)",
  "gpu: requires CUDA",
]
```

- [ ] **Step 2: conda 환경 파일 작성**

`environment-common.yml`:
```yaml
name: disease-detection
channels: [conda-forge, defaults]
dependencies:
  - python=3.11
  - pip
  - pyyaml
  - pillow
  - numpy
  - tqdm
  - pip:
      - python-dotenv
      - hydra-core>=1.3
      - omegaconf>=2.3
      - torchmetrics>=1.4
      - pytorch-lightning>=2.3
      - pytest>=8
      - pytest-mock
      - ruff
      - black
      - -e .
```

`environment-macos.yml`:
```yaml
name: disease-detection
channels: [pytorch, conda-forge, defaults]
dependencies:
  - python=3.11
  - pytorch::pytorch>=2.3
  - pytorch::torchvision>=0.18
  - pip
  - pip:
      - -r environment-common-pip.txt
```

(실용적 단순화로 `environment-common.yml`을 전체 사용, linux는 별도 파일로 cuda 지정)

`environment-linux.yml`:
```yaml
name: disease-detection
channels: [pytorch, nvidia, conda-forge, defaults]
dependencies:
  - python=3.11
  - pytorch::pytorch>=2.3
  - pytorch::torchvision>=0.18
  - pytorch::pytorch-cuda=12.1
  - pip
  - pip:
      - python-dotenv
      - hydra-core>=1.3
      - omegaconf>=2.3
      - torchmetrics>=1.4
      - pytorch-lightning>=2.3
      - pytest>=8
      - pytest-mock
      - ruff
      - black
      - -e .
```

- [ ] **Step 3: `.env.example` 작성**

```bash
# Root for raw images, labels, and intermediate artifacts.
# macmini example:   DATASET_ROOT=/Users/khj/datasets/disease_02
# Ubuntu example:    DATASET_ROOT=/home/user/datasets/disease_02
DATASET_ROOT=

# Ubuntu host for rsync (macmini → Ubuntu). Leave blank on Ubuntu itself.
UBUNTU_USER=
UBUNTU_HOST=
REMOTE_DATASET_ROOT=
```

- [ ] **Step 4: 패키지 디렉토리와 `__init__.py` 생성**

각 디렉토리에 빈 `__init__.py` 생성 (6개: 루트 + utils, data, labeling, models, eval). `src/disease_detection/__init__.py`에는:

```python
"""배·사과 병충해 검출 Phase 1 baseline."""

__version__ = "0.1.0"
```

- [ ] **Step 5: `tests/__init__.py` 빈 파일 + `tests/conftest.py`**

```python
"""공용 pytest 설정."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"
```

- [ ] **Step 6: 설치 및 import 검증**

```bash
conda env create -f environment-macos.yml    # macmini
# 또는
conda env create -f environment-linux.yml    # Ubuntu
conda activate disease-detection
pip install -e .[dev]
python -c "import disease_detection; print(disease_detection.__version__)"
pytest -q
```

Expected: `0.1.0` 출력. pytest는 테스트가 없어도 `no tests ran` 정상 종료.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml environment-*.yml .env.example src/ tests/
git commit -m "Scaffold package, conda env, and pytest skeleton"
```

---

## Task 2: `utils/io.py` — DATASET_ROOT 해석

**Files:**
- Create: `src/disease_detection/utils/io.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_utils.py`:
```python
from __future__ import annotations

import os
from pathlib import Path

import pytest

from disease_detection.utils.io import resolve_dataset_root


def test_resolve_dataset_root_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET_ROOT", str(tmp_path))
    assert resolve_dataset_root() == tmp_path


def test_resolve_dataset_root_raises_when_unset(monkeypatch):
    monkeypatch.delenv("DATASET_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="DATASET_ROOT"):
        resolve_dataset_root()


def test_resolve_dataset_root_expands_user(monkeypatch):
    monkeypatch.setenv("DATASET_ROOT", "~/nonexistent_ds_root_xyz")
    result = resolve_dataset_root(require_exists=False)
    assert result == Path("~/nonexistent_ds_root_xyz").expanduser()


def test_resolve_dataset_root_require_exists(monkeypatch):
    monkeypatch.setenv("DATASET_ROOT", "/definitely/does/not/exist/xyz")
    with pytest.raises(FileNotFoundError):
        resolve_dataset_root(require_exists=True)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_utils.py -v`
Expected: `ImportError` / `ModuleNotFoundError` — `disease_detection.utils.io` not defined.

- [ ] **Step 3: 최소 구현**

`src/disease_detection/utils/io.py`:
```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_utils.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/disease_detection/utils/io.py tests/test_utils.py
git commit -m "Add resolve_dataset_root util with env-based resolution"
```

---

## Task 3: `utils/seeding.py` — 재현성 시드

**Files:**
- Create: `src/disease_detection/utils/seeding.py`
- Modify: `tests/test_utils.py` (append)

- [ ] **Step 1: 실패 테스트 작성 (append)**

`tests/test_utils.py` 말미에:
```python
import random

import numpy as np
import torch

from disease_detection.utils.seeding import set_seed


def test_set_seed_makes_random_deterministic():
    set_seed(123)
    a = [random.random(), np.random.rand(), float(torch.rand(1))]
    set_seed(123)
    b = [random.random(), np.random.rand(), float(torch.rand(1))]
    assert a == b


def test_set_seed_different_values_differ():
    set_seed(1)
    a = float(torch.rand(1))
    set_seed(2)
    b = float(torch.rand(1))
    assert a != b
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_utils.py::test_set_seed_makes_random_deterministic -v`
Expected: ImportError on `set_seed`.

- [ ] **Step 3: 구현**

`src/disease_detection/utils/seeding.py`:
```python
"""재현성을 위한 일괄 seed 설정."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Python random / numpy / torch / PYTHONHASHSEED 동시 시드.

    Lightning의 `seed_everything`과 같은 역할을 하지만 의존성 최소화를 위해 직접 구현.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_utils.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/disease_detection/utils/seeding.py tests/test_utils.py
git commit -m "Add set_seed util for reproducible random state"
```

---

## Task 4: `data/transforms.py` — torchvision v2 augmentation

**Files:**
- Create: `src/disease_detection/data/transforms.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_data.py`:
```python
from __future__ import annotations

import torch

from disease_detection.data.transforms import (
    build_classifier_train_transform,
    build_classifier_eval_transform,
    build_detector_train_transform,
    build_detector_eval_transform,
)


def test_classifier_train_transform_output_shape():
    tfm = build_classifier_train_transform()
    img = torch.randint(0, 255, (3, 512, 640), dtype=torch.uint8)
    out = tfm(img)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_classifier_eval_transform_output_shape():
    tfm = build_classifier_eval_transform()
    img = torch.randint(0, 255, (3, 512, 640), dtype=torch.uint8)
    out = tfm(img)
    assert out.shape == (3, 224, 224)


def test_detector_train_transform_preserves_bbox_count():
    tfm = build_detector_train_transform()
    img = torch.randint(0, 255, (3, 400, 400), dtype=torch.uint8)
    # torchvision v2 detection transforms use tv_tensors
    from torchvision import tv_tensors

    boxes = tv_tensors.BoundingBoxes(
        [[10, 10, 100, 100], [50, 50, 200, 200]],
        format="XYXY",
        canvas_size=(400, 400),
    )
    target = {"boxes": boxes, "labels": torch.tensor([1, 2])}
    img_out, target_out = tfm(img, target)
    assert len(target_out["boxes"]) == 2
    assert img_out.dtype == torch.float32
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_data.py -v`
Expected: ImportError.

- [ ] **Step 3: 구현**

`src/disease_detection/data/transforms.py`:
```python
"""torchvision v2 기반 변환 생성기.

분류기는 단일 텐서, detector는 이미지+target 쌍을 다룸. v2는 두 경우 모두 동일 API.
"""
from __future__ import annotations

from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_classifier_train_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(224, scale=(0.7, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.RandomErasing(p=0.25, scale=(0.02, 0.1)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_classifier_eval_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256, antialias=True),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_detector_train_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def build_detector_eval_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


# `torch`를 위에서 사용하기 위한 import (v2 내부에서 필요)
import torch  # noqa: E402
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_data.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/disease_detection/data/transforms.py tests/test_data.py
git commit -m "Add torchvision v2 transforms for classifier and detector"
```

---

## Task 5: `data/aihub.py` — AIhub 파서

**배경:** AIhub 어노테이션의 정확한 스키마는 다운로드 전엔 알 수 없으므로, 파서는 **JSON 기반으로 가정**한 참조 구현 + pluggable format loader 구조로 작성. 실제 스키마가 달라지면 loader 함수만 교체.

**Files:**
- Create: `src/disease_detection/data/aihub.py`
- Create: `tests/fixtures/dummy_aihub/images/sample1.jpg`
- Create: `tests/fixtures/dummy_aihub/annotations/sample1.json`
- Modify: `tests/test_data.py` (append)

- [ ] **Step 1: 테스트 fixture 준비**

```bash
mkdir -p tests/fixtures/dummy_aihub/{images,annotations}
python - <<'EOF'
from PIL import Image
from pathlib import Path
import json

img_path = Path("tests/fixtures/dummy_aihub/images/sample1.jpg")
Image.new("RGB", (640, 480), color=(120, 200, 90)).save(img_path)

ann = {
    "image": {"file_name": "sample1.jpg", "width": 640, "height": 480},
    "annotations": [
        {"bbox": [100, 80, 250, 300], "category": "leaf"},
        {"bbox": [300, 200, 420, 360], "category": "fruit"},
    ],
    "labels": {"fireblight": 1},
    "meta": {"crop": "pear"},
}
Path("tests/fixtures/dummy_aihub/annotations/sample1.json").write_text(
    json.dumps(ann), encoding="utf-8"
)
EOF
```

- [ ] **Step 2: 실패 테스트 작성 (`tests/test_data.py`에 append)**

```python
from pathlib import Path

from disease_detection.data.aihub import AIhubImage, load_aihub_split


def test_load_aihub_split_parses_fixture(fixtures_dir):
    root = fixtures_dir / "dummy_aihub"
    entries = load_aihub_split(root)
    assert len(entries) == 1
    entry = entries[0]
    assert isinstance(entry, AIhubImage)
    assert entry.crop == "pear"
    assert entry.fireblight == 1
    assert entry.width == 640
    assert entry.height == 480
    assert len(entry.boxes) == 2
    assert entry.boxes[0].category == "leaf"
    assert entry.boxes[0].xyxy == (100.0, 80.0, 350.0, 380.0)  # x+w, y+h
```

- [ ] **Step 3: 실패 확인**

Run: `pytest tests/test_data.py::test_load_aihub_split_parses_fixture -v`
Expected: ImportError.

- [ ] **Step 4: 구현**

`src/disease_detection/data/aihub.py`:
```python
"""AIhub 과수원 화상병 데이터셋 파서.

실제 AIhub 배포 스키마가 확정되기 전까진 JSON 포맷을 가정. 스키마가 달라지면
`_load_annotation_json` 을 교체하거나 format-specific loader를 추가.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AIhubBox:
    """단일 부위 bbox + 카테고리."""

    category: str  # leaf / stem / fruit
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class AIhubImage:
    """한 장 이미지의 완전한 어노테이션."""

    image_path: Path
    crop: str  # "pear" or "apple"
    width: int
    height: int
    fireblight: int  # AIhub 원본 이진 라벨 (0 / 1)
    boxes: tuple[AIhubBox, ...] = field(default_factory=tuple)


def _load_annotation_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_entry(image_path: Path, ann: dict) -> AIhubImage:
    img_meta = ann["image"]
    boxes: list[AIhubBox] = []
    for a in ann.get("annotations", []):
        x, y, w, h = a["bbox"]
        boxes.append(
            AIhubBox(
                category=a["category"],
                xyxy=(float(x), float(y), float(x + w), float(y + h)),
            )
        )
    return AIhubImage(
        image_path=image_path,
        crop=ann.get("meta", {}).get("crop", "unknown"),
        width=int(img_meta["width"]),
        height=int(img_meta["height"]),
        fireblight=int(ann.get("labels", {}).get("fireblight", 0)),
        boxes=tuple(boxes),
    )


def load_aihub_split(root: Path) -> list[AIhubImage]:
    """`root/images/*.jpg` + `root/annotations/*.json` 쌍으로부터 항목 수집.

    어노테이션 파일 이름은 이미지 파일의 stem과 동일해야 한다.
    """
    images_dir = root / "images"
    ann_dir = root / "annotations"
    entries: list[AIhubImage] = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        ann_path = ann_dir / f"{img_path.stem}.json"
        if not ann_path.exists():
            continue
        ann = _load_annotation_json(ann_path)
        entries.append(_parse_entry(img_path, ann))
    return entries
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `pytest tests/test_data.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/disease_detection/data/aihub.py tests/fixtures/ tests/test_data.py
git commit -m "Add AIhub parser with dataclass schema and JSON loader"
```

---

## Task 6: `data/detection_dataset.py` — Faster R-CNN용 Dataset

**Files:**
- Create: `src/disease_detection/data/detection_dataset.py`
- Modify: `tests/test_data.py` (append)

- [ ] **Step 1: 실패 테스트 작성**

```python
from disease_detection.data.detection_dataset import DetectionDataset, PART_CATEGORIES


def test_part_categories_mapping():
    # background=0, leaf=1, stem=2, fruit=3
    assert PART_CATEGORIES["leaf"] == 1
    assert PART_CATEGORIES["stem"] == 2
    assert PART_CATEGORIES["fruit"] == 3


def test_detection_dataset_item_shapes(fixtures_dir):
    ds = DetectionDataset.from_aihub_root(fixtures_dir / "dummy_aihub")
    img, target = ds[0]
    assert img.ndim == 3 and img.shape[0] == 3  # CHW
    assert target["boxes"].shape == (2, 4)
    assert target["labels"].tolist() == [1, 3]  # leaf, fruit
    assert target["image_id"].numel() == 1
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_data.py -v` → ImportError.

- [ ] **Step 3: 구현**

`src/disease_detection/data/detection_dataset.py`:
```python
"""Faster R-CNN 학습·추론용 Dataset.

torchvision detection reference 규격을 따른다: `target` = dict with `boxes`, `labels`,
`image_id`, `area`, `iscrowd`. v2 transforms는 `tv_tensors.BoundingBoxes` 를 처리.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors

from .aihub import AIhubImage, load_aihub_split

PART_CATEGORIES: dict[str, int] = {"leaf": 1, "stem": 2, "fruit": 3}


class DetectionDataset(Dataset):
    def __init__(
        self,
        entries: list[AIhubImage],
        transform: Callable | None = None,
    ) -> None:
        self.entries = entries
        self.transform = transform

    @classmethod
    def from_aihub_root(
        cls, root: Path, transform: Callable | None = None
    ) -> "DetectionDataset":
        return cls(load_aihub_split(root), transform=transform)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        pil = Image.open(entry.image_path).convert("RGB")
        img = tv_tensors.Image(pil)

        if entry.boxes:
            xyxy = torch.tensor([b.xyxy for b in entry.boxes], dtype=torch.float32)
            labels = torch.tensor(
                [PART_CATEGORIES[b.category] for b in entry.boxes], dtype=torch.int64
            )
        else:
            xyxy = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        boxes = tv_tensors.BoundingBoxes(
            xyxy, format="XYXY", canvas_size=(entry.height, entry.width)
        )
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": areas,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_data.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/disease_detection/data/detection_dataset.py tests/test_data.py
git commit -m "Add DetectionDataset producing torchvision detection targets"
```

---

## Task 7: `data/classification_dataset.py` — 분류기용 crop Dataset

**Files:**
- Create: `src/disease_detection/data/classification_dataset.py`
- Create: `tests/fixtures/dummy_vlm_labels.jsonl`
- Modify: `tests/test_data.py` (append)

- [ ] **Step 1: VLM 라벨 fixture 작성**

```bash
python - <<'EOF'
import json
from pathlib import Path
line = {
    "image_path": "images/sample1.jpg",
    "image_sha256": "dummy",
    "crop": "pear",
    "plant_part": "leaf",
    "classification": "DEFECT",
    "severity": 6,
    "explanation": "leaf tip browning",
    "model": "claude-haiku-4-5",
    "prompt_version": "v1",
    "timestamp": "2026-04-21T00:00:00Z",
}
Path("tests/fixtures/dummy_vlm_labels.jsonl").write_text(json.dumps(line) + "\n", encoding="utf-8")
EOF
```

- [ ] **Step 2: 실패 테스트 작성**

```python
from disease_detection.data.classification_dataset import (
    build_fireblight_crops,
    build_defect_crops,
    ClassificationCropDataset,
)


def test_build_fireblight_crops_uses_aihub_labels(fixtures_dir):
    items = build_fireblight_crops(fixtures_dir / "dummy_aihub")
    # sample1 has fireblight=1, 2 boxes → 2 crop items, both label=1
    assert len(items) == 2
    assert all(item.label == 1 for item in items)
    assert {item.plant_part for item in items} == {"leaf", "fruit"}


def test_build_defect_crops_applies_severity_threshold(fixtures_dir):
    items = build_defect_crops(
        fixtures_dir / "dummy_aihub",
        fixtures_dir / "dummy_vlm_labels.jsonl",
        defect_threshold=4,
    )
    # severity=6 → defect (label=1). Only leaf box matches vlm plant_part.
    assert len(items) >= 1
    assert any(item.label == 1 and item.plant_part == "leaf" for item in items)


def test_classification_crop_dataset_returns_tensor(fixtures_dir):
    items = build_fireblight_crops(fixtures_dir / "dummy_aihub")
    ds = ClassificationCropDataset(items)
    img, label = ds[0]
    assert img.ndim == 3 and img.shape[0] == 3
    assert label in (0, 1)
```

- [ ] **Step 3: 실패 확인**

Run: `pytest tests/test_data.py -v` → ImportError.

- [ ] **Step 4: 구현**

`src/disease_detection/data/classification_dataset.py`:
```python
"""ResNet18 학습·추론용 분류기 crop Dataset.

화상병 분류기와 범용 결함 분류기가 같은 Dataset 클래스를 공유. 라벨 생성 함수만 분리:
- `build_fireblight_crops` — AIhub 원본 이진 라벨 기반, 모든 bbox에 이미지 단위 라벨 적용.
- `build_defect_crops` — VLM severity JSONL을 이미지·부위별로 매칭, 임계값으로 이진화.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors

from .aihub import AIhubBox, AIhubImage, load_aihub_split


@dataclass(frozen=True)
class CropItem:
    """단일 분류기 학습 아이템."""

    image_path: Path
    xyxy: tuple[float, float, float, float]
    plant_part: str
    crop: str
    label: int  # 0 or 1


def build_fireblight_crops(aihub_root: Path) -> list[CropItem]:
    entries = load_aihub_split(aihub_root)
    items: list[CropItem] = []
    for entry in entries:
        for box in entry.boxes:
            items.append(
                CropItem(
                    image_path=entry.image_path,
                    xyxy=box.xyxy,
                    plant_part=box.category,
                    crop=entry.crop,
                    label=int(entry.fireblight),
                )
            )
    return items


def build_defect_crops(
    aihub_root: Path,
    vlm_labels_jsonl: Path,
    defect_threshold: int = 4,
) -> list[CropItem]:
    """VLM JSONL을 `(image_stem, plant_part) → severity` 맵으로 변환 후 매칭.

    이미지+부위 조합에 VLM 라벨이 있는 경우만 CropItem 생성. 없는 조합은 스킵.
    """
    sev_map: dict[tuple[str, str], int] = {}
    with vlm_labels_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            stem = Path(obj["image_path"]).stem
            part = obj["plant_part"]
            sev_map[(stem, part)] = int(obj["severity"])

    entries = load_aihub_split(aihub_root)
    items: list[CropItem] = []
    for entry in entries:
        for box in entry.boxes:
            key = (entry.image_path.stem, box.category)
            if key not in sev_map:
                continue
            sev = sev_map[key]
            label = 1 if sev >= defect_threshold else 0
            items.append(
                CropItem(
                    image_path=entry.image_path,
                    xyxy=box.xyxy,
                    plant_part=box.category,
                    crop=entry.crop,
                    label=label,
                )
            )
    return items


class ClassificationCropDataset(Dataset):
    """CropItem 리스트를 Tensor로 로딩."""

    def __init__(
        self,
        items: list[CropItem],
        transform: Callable | None = None,
    ) -> None:
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        pil = Image.open(item.image_path).convert("RGB")
        x1, y1, x2, y2 = item.xyxy
        crop = pil.crop((x1, y1, x2, y2))
        img = tv_tensors.Image(crop)
        if self.transform is not None:
            img = self.transform(img)
        else:
            # 기본: uint8 tensor 그대로 반환
            img = torch.as_tensor(list(crop.getdata()), dtype=torch.uint8).reshape(
                crop.height, crop.width, 3
            ).permute(2, 0, 1)
        return img, int(item.label)
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `pytest tests/test_data.py -v`
Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git add src/disease_detection/data/classification_dataset.py tests/fixtures/dummy_vlm_labels.jsonl tests/test_data.py
git commit -m "Add ClassificationCropDataset with fireblight and defect builders"
```

---

## Task 8: `labeling/prompts.py` — VLM 심각도 프롬프트

**Files:**
- Create: `src/disease_detection/labeling/prompts.py`
- Create: `tests/test_labeling.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_labeling.py`:
```python
from __future__ import annotations

from disease_detection.labeling.prompts import SEVERITY_PROMPT_V1, PROMPT_VERSION


def test_prompt_contains_pptx_keywords():
    # Keys from PPT Slide 2
    assert "NORMAL" in SEVERITY_PROMPT_V1
    assert "DEFECT" in SEVERITY_PROMPT_V1
    assert "severity" in SEVERITY_PROMPT_V1.lower()


def test_prompt_asks_for_structured_format():
    assert "Classification" in SEVERITY_PROMPT_V1
    assert "Severity" in SEVERITY_PROMPT_V1


def test_prompt_version_string():
    assert PROMPT_VERSION == "v1"
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_labeling.py -v` → ImportError.

- [ ] **Step 3: 구현**

`src/disease_detection/labeling/prompts.py`:
```python
"""VLM 재라벨링용 프롬프트. PPT Slide 2 원문을 충실히 반영.

프롬프트 수정 시 반드시 `PROMPT_VERSION` 을 올리고, 기존 버전 상수를 보존.
JSONL 라벨의 `prompt_version` 필드와 매칭되어 라벨 이력 추적에 사용됨.
"""
from __future__ import annotations

PROMPT_VERSION: str = "v1"

SEVERITY_PROMPT_V1: str = (
    "Inspect this orchard image carefully. Evaluate the condition of the visible plant part.\n\n"
    "If the plant part appears completely fresh, intact, and undamaged—no visible signs of "
    "drying, pest damage, holes, discoloration, rot, deformities, mold, or breakage—classify "
    "the image as **NORMAL**.\n\n"
    "If any defects are present, classify the image as **DEFECT**. Then, assign a "
    "**severity score from 1 to 10**, where:\n"
    "- **1** = Very minor, cosmetic, or negligible defect (e.g., a small dry spot)\n"
    "- **5** = Moderate defect that affects part of the plant\n"
    "- **10** = Severe, widespread, or critical damage\n\n"
    "### Return the following format as a single JSON object (no prose):\n"
    '```json\n{"classification": "NORMAL|DEFECT", "severity": 0-10, "explanation": "..."}\n```\n\n'
    "### Examples:\n"
    '- {"classification": "NORMAL", "severity": 0, "explanation": "fruit is healthy"}\n'
    '- {"classification": "DEFECT", "severity": 2, "explanation": "minor leaf tip browning"}\n'
    '- {"classification": "DEFECT", "severity": 8, "explanation": "fruit is partially rotten"}\n'
)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_labeling.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/disease_detection/labeling/prompts.py tests/test_labeling.py
git commit -m "Add severity prompt v1 based on PPT slide 2"
```

---

## Task 9: `labeling/vlm_client.py` — Claude Code CLI subprocess 래퍼

**Files:**
- Create: `src/disease_detection/labeling/vlm_client.py`
- Modify: `tests/test_labeling.py` (append)

- [ ] **Step 1: 실패 테스트 작성**

```python
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from disease_detection.labeling.vlm_client import (
    VLMLabel,
    call_claude_cli,
    parse_vlm_response,
)


def test_parse_vlm_response_valid_json():
    raw = '{"classification": "DEFECT", "severity": 6, "explanation": "browning"}'
    label = parse_vlm_response(raw)
    assert label == VLMLabel(classification="DEFECT", severity=6, explanation="browning")


def test_parse_vlm_response_with_code_fences():
    raw = '```json\n{"classification": "NORMAL", "severity": 0, "explanation": "ok"}\n```'
    label = parse_vlm_response(raw)
    assert label.severity == 0


def test_parse_vlm_response_invalid_raises():
    import pytest

    with pytest.raises(ValueError):
        parse_vlm_response("not json at all")


def test_call_claude_cli_invokes_subprocess(mocker, tmp_path):
    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = '{"classification": "DEFECT", "severity": 5, "explanation": "x"}'
    completed.stderr = ""
    run_mock = mocker.patch(
        "disease_detection.labeling.vlm_client.subprocess.run",
        return_value=completed,
    )

    img = tmp_path / "a.jpg"
    img.write_bytes(b"fake")

    label = call_claude_cli(img, prompt="P", model="haiku")

    assert label.severity == 5
    args, kwargs = run_mock.call_args
    assert args[0][0] == "claude"
    assert "-p" in args[0]
    assert "--model" in args[0]
    assert "haiku" in args[0]
    assert str(img) in " ".join(args[0])


def test_call_claude_cli_nonzero_exit_raises(mocker, tmp_path):
    import pytest
    completed = MagicMock(returncode=1, stdout="", stderr="rate limit")
    mocker.patch(
        "disease_detection.labeling.vlm_client.subprocess.run",
        return_value=completed,
    )
    img = tmp_path / "a.jpg"
    img.write_bytes(b"fake")
    with pytest.raises(RuntimeError, match="rate limit"):
        call_claude_cli(img, prompt="P", model="haiku")
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_labeling.py -v` → ImportError.

- [ ] **Step 3: 구현**

`src/disease_detection/labeling/vlm_client.py`:
```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_labeling.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/disease_detection/labeling/vlm_client.py tests/test_labeling.py
git commit -m "Add Claude CLI subprocess wrapper with JSON response parser"
```

---

## Task 10: `labeling/batch_label.py` — 재개 가능 배치 라벨링

**Files:**
- Create: `src/disease_detection/labeling/batch_label.py`
- Modify: `tests/test_labeling.py` (append)

- [ ] **Step 1: 실패 테스트 작성**

```python
from disease_detection.labeling.batch_label import (
    BatchJob,
    BatchResult,
    hash_image_file,
    load_completed_hashes,
    run_batch,
)


def test_hash_image_file_deterministic(tmp_path):
    p = tmp_path / "x.jpg"
    p.write_bytes(b"hello world")
    h1 = hash_image_file(p)
    h2 = hash_image_file(p)
    assert h1 == h2 and len(h1) == 64  # sha256 hex


def test_load_completed_hashes_reads_existing_jsonl(tmp_path):
    jsonl = tmp_path / "labels.jsonl"
    jsonl.write_text(
        '{"image_sha256": "a", "classification": "NORMAL"}\n'
        '{"image_sha256": "b", "classification": "DEFECT"}\n',
        encoding="utf-8",
    )
    assert load_completed_hashes(jsonl) == {"a", "b"}


def test_run_batch_skips_existing_hash(tmp_path, mocker):
    from disease_detection.labeling.vlm_client import VLMLabel

    img1 = tmp_path / "a.jpg"
    img1.write_bytes(b"one")
    img2 = tmp_path / "b.jpg"
    img2.write_bytes(b"two")

    jsonl = tmp_path / "out.jsonl"
    # a.jpg is pre-labeled
    jsonl.write_text(
        '{"image_sha256": "' + hash_image_file(img1) + '", "severity": 0}\n',
        encoding="utf-8",
    )

    call_mock = mocker.patch(
        "disease_detection.labeling.batch_label.call_claude_cli",
        return_value=VLMLabel(classification="DEFECT", severity=7, explanation="x"),
    )

    jobs = [
        BatchJob(image_path=img1, crop="pear", plant_part="leaf"),
        BatchJob(image_path=img2, crop="pear", plant_part="leaf"),
    ]
    result: BatchResult = run_batch(
        jobs=jobs,
        output_jsonl=jsonl,
        prompt="P",
        prompt_version="v1",
        model="haiku",
    )

    assert result.processed == 1  # only b.jpg
    assert result.skipped == 1
    assert call_mock.call_count == 1


def test_run_batch_writes_errors(tmp_path, mocker):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"one")
    jsonl = tmp_path / "out.jsonl"

    mocker.patch(
        "disease_detection.labeling.batch_label.call_claude_cli",
        side_effect=RuntimeError("rate limit"),
    )
    # minimize backoff for test
    mocker.patch("disease_detection.labeling.batch_label.time.sleep", return_value=None)

    jobs = [BatchJob(image_path=img, crop="pear", plant_part="leaf")]
    result = run_batch(
        jobs=jobs,
        output_jsonl=jsonl,
        prompt="P",
        prompt_version="v1",
        model="haiku",
        max_retries=2,
    )
    assert result.failed == 1
    errors_path = jsonl.with_name(jsonl.stem + ".errors.jsonl")
    assert errors_path.exists()
    assert "rate limit" in errors_path.read_text()
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_labeling.py -v` → ImportError.

- [ ] **Step 3: 구현**

`src/disease_detection/labeling/batch_label.py`:
```python
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
    image_path: Path
    crop: str
    plant_part: str


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
    record = {
        "image_path": str(job.image_path),
        "image_sha256": sha,
        "crop": job.crop,
        "plant_part": job.plant_part,
        "classification": label.classification,
        "severity": label.severity,
        "explanation": label.explanation,
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
                    "plant_part": job.plant_part,
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_labeling.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/disease_detection/labeling/batch_label.py tests/test_labeling.py
git commit -m "Add resume-capable VLM batch labeling runner with retry + error log"
```

---

## Task 11: `models/detector.py` — Faster R-CNN LightningModule

**Files:**
- Create: `src/disease_detection/models/detector.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_models.py`:
```python
from __future__ import annotations

import torch

from disease_detection.models.detector import FasterRCNNModule


def test_detector_module_forward_smoke():
    module = FasterRCNNModule(num_classes=4)
    module.eval()
    images = [torch.rand(3, 320, 320)]
    with torch.no_grad():
        outputs = module(images)
    assert len(outputs) == 1
    assert set(outputs[0].keys()) >= {"boxes", "labels", "scores"}


def test_detector_training_step_returns_loss():
    module = FasterRCNNModule(num_classes=4)
    module.train()
    images = [torch.rand(3, 256, 256)]
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0]]),
            "labels": torch.tensor([1]),
        }
    ]
    loss = module.training_step((images, targets), batch_idx=0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_models.py -v` → ImportError.

- [ ] **Step 3: 구현**

`src/disease_detection/models/detector.py`:
```python
"""torchvision Faster R-CNN ResNet50-FPN v2 LightningModule."""
from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNNModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 4,  # background + leaf + stem + fruit
        lr: float = 0.005,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        lr_step: int = 10,
        lr_gamma: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model = model

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())
        self.log_dict(
            {f"train/{k}": v for k, v in loss_dict.items()},
            on_step=True, on_epoch=True, prog_bar=False,
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # torchvision detection heads return dict of losses in train, predictions in eval.
        # Use a separate train()/eval() split if you want both; here: predictions only.
        self.model.eval()
        with torch.no_grad():
            preds = self.model(images)
        self.model.train()
        # mAP는 epoch 말미에 집계 — 여기선 prediction만 축적
        if not hasattr(self, "_val_cache"):
            self._val_cache = []
        self._val_cache.append((preds, targets))

    def on_validation_epoch_end(self) -> None:
        # 실제 mAP 계산은 evaluate 스크립트에서. 여기선 샘플 수만 로깅하여 sanity check.
        count = len(getattr(self, "_val_cache", []))
        self.log("val/batches", float(count), prog_bar=False)
        self._val_cache = []

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = SGD(
            params,
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = StepLR(
            optimizer,
            step_size=self.hparams.lr_step,
            gamma=self.hparams.lr_gamma,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_models.py -v`
Expected: 2 passed (최초 모델 다운로드가 일어날 수 있어 시간 소요).

- [ ] **Step 5: Commit**

```bash
git add src/disease_detection/models/detector.py tests/test_models.py
git commit -m "Add FasterRCNNModule with ResNet50-FPN v2 backbone"
```

---

## Task 12: `models/classifier.py` — ResNet18 이진 분류기 LightningModule

**Files:**
- Create: `src/disease_detection/models/classifier.py`
- Modify: `tests/test_models.py` (append)

- [ ] **Step 1: 실패 테스트 작성**

```python
from disease_detection.models.classifier import PlantDefectClassifier


def test_classifier_forward_shape():
    module = PlantDefectClassifier()
    module.eval()
    x = torch.rand(4, 3, 224, 224)
    with torch.no_grad():
        logits = module(x)
    assert logits.shape == (4,)


def test_classifier_training_step_returns_loss():
    module = PlantDefectClassifier()
    x = torch.rand(2, 3, 224, 224)
    y = torch.tensor([0, 1])
    loss = module.training_step((x, y), batch_idx=0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_classifier_overfits_two_samples():
    """10-step overfit sanity check — loss must decrease."""
    torch.manual_seed(0)
    module = PlantDefectClassifier(lr=1e-2)
    x = torch.rand(2, 3, 224, 224)
    y = torch.tensor([0, 1])
    opt = module.configure_optimizers()["optimizer"]
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = module.training_step((x, y), batch_idx=0)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] * 0.5
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_models.py -v` → ImportError.

- [ ] **Step 3: 구현**

`src/disease_detection/models/classifier.py`:
```python
"""ResNet18 기반 이진 분류기 LightningModule.

화상병용·범용 결함용으로 두 인스턴스를 독립 학습. 구조는 동일, 학습 데이터·체크포인트만 분리.
"""
from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import ResNet18_Weights, resnet18


class PlantDefectClassifier(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        pos_weight: float | None = None,
        t_max: int = 30,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 1)
        self.model = backbone

        pw = (
            torch.tensor([pos_weight], dtype=torch.float32)
            if pos_weight is not None
            else None
        )
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        preds = (torch.sigmoid(logits) >= 0.5).long()
        acc = (preds == y).float().mean()
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.t_max)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_models.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/disease_detection/models/classifier.py tests/test_models.py
git commit -m "Add PlantDefectClassifier ResNet18 binary module with overfit sanity"
```

---

## Task 13: `models/pipeline.py` — 2단계 추론 래퍼

**Files:**
- Create: `src/disease_detection/models/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_pipeline.py`:
```python
from __future__ import annotations

import torch
from PIL import Image

from disease_detection.models.pipeline import TwoStagePipeline, PipelinePrediction


class _FakeDetector:
    def __call__(self, images):
        return [
            {
                "boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0]]),
                "labels": torch.tensor([1]),  # leaf
                "scores": torch.tensor([0.9]),
            }
        ]


class _FakeClassifier:
    def __init__(self, prob: float) -> None:
        self.prob = prob

    def __call__(self, crops):
        batch = crops.shape[0]
        return torch.logit(torch.full((batch,), self.prob))


def test_pipeline_runs_end_to_end(tmp_path):
    img = Image.new("RGB", (200, 200), color=(200, 50, 50))
    img_path = tmp_path / "x.jpg"
    img.save(img_path)

    pipe = TwoStagePipeline(
        detector=_FakeDetector(),
        fireblight_classifier=_FakeClassifier(prob=0.8),
        defect_classifier=_FakeClassifier(prob=0.3),
        score_threshold=0.5,
    )
    result = pipe.predict_image(img_path, crop="pear")
    assert isinstance(result, PipelinePrediction)
    assert len(result.detections) == 1
    det = result.detections[0]
    assert det.plant_part == "leaf"
    assert det.fireblight_prob > 0.5
    assert det.defect_prob < 0.5


def test_pipeline_filters_by_score_threshold(tmp_path):
    class LowScoreDetector:
        def __call__(self, images):
            return [
                {
                    "boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0]]),
                    "labels": torch.tensor([1]),
                    "scores": torch.tensor([0.1]),
                }
            ]

    img_path = tmp_path / "x.jpg"
    Image.new("RGB", (120, 120)).save(img_path)

    pipe = TwoStagePipeline(
        detector=LowScoreDetector(),
        fireblight_classifier=_FakeClassifier(prob=0.9),
        defect_classifier=_FakeClassifier(prob=0.9),
        score_threshold=0.5,
    )
    result = pipe.predict_image(img_path, crop="pear")
    assert result.detections == []
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_pipeline.py -v` → ImportError.

- [ ] **Step 3: 구현**

`src/disease_detection/models/pipeline.py`:
```python
"""Detector + 두 분류기 결합 추론 래퍼.

Detector는 부위 bbox를 예측하고, 각 bbox를 crop하여 두 분류기 각각에 입력.
`fireblight_prob`과 `defect_prob`을 동시에 반환.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2

from ..data.detection_dataset import PART_CATEGORIES
from ..data.transforms import (
    build_classifier_eval_transform,
    build_detector_eval_transform,
)

_INV_PART = {v: k for k, v in PART_CATEGORIES.items()}


class _DetectorLike(Protocol):
    def __call__(self, images: list[torch.Tensor]) -> list[dict]: ...


class _ClassifierLike(Protocol):
    def __call__(self, crops: torch.Tensor) -> torch.Tensor: ...


@dataclass
class Detection:
    plant_part: str
    xyxy: tuple[float, float, float, float]
    score: float
    fireblight_prob: float
    defect_prob: float


@dataclass
class PipelinePrediction:
    image_path: Path
    crop: str
    detections: list[Detection] = field(default_factory=list)


class TwoStagePipeline:
    def __init__(
        self,
        detector: _DetectorLike,
        fireblight_classifier: _ClassifierLike,
        defect_classifier: _ClassifierLike,
        score_threshold: float = 0.5,
        detector_transform: v2.Compose | None = None,
        classifier_transform: v2.Compose | None = None,
    ) -> None:
        self.detector = detector
        self.fireblight = fireblight_classifier
        self.defect = defect_classifier
        self.score_threshold = score_threshold
        self.det_tfm = detector_transform or build_detector_eval_transform()
        self.cls_tfm = classifier_transform or build_classifier_eval_transform()

    def predict_image(self, image_path: Path, crop: str) -> PipelinePrediction:
        pil = Image.open(image_path).convert("RGB")
        det_input = self.det_tfm(tv_tensors.Image(pil))
        outputs = self.detector([det_input])[0]

        boxes = outputs["boxes"]
        labels = outputs["labels"]
        scores = outputs["scores"]

        keep = scores >= self.score_threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]

        detections: list[Detection] = []
        if len(boxes) == 0:
            return PipelinePrediction(image_path=image_path, crop=crop, detections=[])

        # crop + 분류기 배치 입력
        crops: list[torch.Tensor] = []
        for xyxy in boxes.tolist():
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
            c = pil.crop((x1, y1, x2, y2))
            crops.append(self.cls_tfm(tv_tensors.Image(c)))
        crop_batch = torch.stack(crops)

        fire_probs = torch.sigmoid(self.fireblight(crop_batch)).tolist()
        defect_probs = torch.sigmoid(self.defect(crop_batch)).tolist()

        for box, label_id, score, fp, dp in zip(
            boxes.tolist(), labels.tolist(), scores.tolist(), fire_probs, defect_probs
        ):
            detections.append(
                Detection(
                    plant_part=_INV_PART.get(int(label_id), "unknown"),
                    xyxy=tuple(float(v) for v in box),
                    score=float(score),
                    fireblight_prob=float(fp),
                    defect_prob=float(dp),
                )
            )
        return PipelinePrediction(
            image_path=image_path, crop=crop, detections=detections
        )
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_pipeline.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/disease_detection/models/pipeline.py tests/test_pipeline.py
git commit -m "Add TwoStagePipeline combining detector with both classifiers"
```

---

## Task 14: `eval/metrics.py` — torchmetrics 래퍼

**Files:**
- Create: `src/disease_detection/eval/metrics.py`
- Create: `tests/test_eval.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_eval.py`:
```python
from __future__ import annotations

import torch

from disease_detection.eval.metrics import (
    ClassificationMetricsReport,
    compute_classification_report,
    compute_detection_map,
)


def test_compute_classification_report_perfect():
    y_true = torch.tensor([0, 1, 1, 0, 1])
    y_score = torch.tensor([0.1, 0.9, 0.8, 0.2, 0.95])
    rep = compute_classification_report(y_true, y_score, threshold=0.5)
    assert isinstance(rep, ClassificationMetricsReport)
    assert rep.accuracy == 1.0
    assert rep.recall == 1.0
    assert rep.precision == 1.0
    assert rep.f1 == 1.0


def test_compute_classification_report_recall_at_precision():
    # Scores designed so that tuning threshold matters.
    y_true = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    y_score = torch.tensor([0.1, 0.2, 0.6, 0.4, 0.55, 0.7, 0.9])
    rep = compute_classification_report(y_true, y_score, threshold=0.5)
    # Recall@Precision>=0.7 should be defined (not None) for this distribution.
    assert rep.recall_at_precision_70 is not None


def test_compute_detection_map_smoke():
    preds = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.9]),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([1]),
        }
    ]
    m = compute_detection_map(preds, targets)
    assert "map_50" in m
    assert m["map_50"] >= 0.99
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_eval.py -v` → ImportError.

- [ ] **Step 3: 구현**

`src/disease_detection/eval/metrics.py`:
```python
"""torchmetrics 기반 classifier·detector 평가 리포트."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryPrecision,
    BinaryPrecisionRecallCurve,
    BinaryRecall,
    BinaryROC,
)
from torchmetrics.detection import MeanAveragePrecision


@dataclass
class ClassificationMetricsReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    recall_at_precision_70: float | None
    confusion_matrix: list[list[int]]


def compute_classification_report(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    threshold: float = 0.5,
    target_precision: float = 0.7,
) -> ClassificationMetricsReport:
    y_true = y_true.long()
    preds = (y_score >= threshold).long()

    acc = float(BinaryAccuracy()(preds, y_true))
    prec = float(BinaryPrecision()(preds, y_true))
    rec = float(BinaryRecall()(preds, y_true))
    f1 = float(BinaryF1Score()(preds, y_true))
    cm = BinaryConfusionMatrix()(preds, y_true).tolist()

    from torchmetrics.classification import BinaryAUROC

    try:
        roc_auc = float(BinaryAUROC()(y_score, y_true))
    except Exception:
        roc_auc = None

    # Recall@Precision>=target_precision
    pr = BinaryPrecisionRecallCurve()
    pr.update(y_score, y_true)
    precisions, recalls, _ = pr.compute()
    mask = precisions >= target_precision
    recall_at_p = float(recalls[mask].max()) if mask.any() else None

    return ClassificationMetricsReport(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc_auc,
        recall_at_precision_70=recall_at_p,
        confusion_matrix=cm,
    )


def compute_detection_map(
    preds: list[dict],
    targets: list[dict],
) -> dict[str, float]:
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=False)
    metric.update(preds, targets)
    raw = metric.compute()
    return {
        "map": float(raw["map"]),
        "map_50": float(raw["map_50"]),
        "map_75": float(raw["map_75"]),
    }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_eval.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/disease_detection/eval/metrics.py tests/test_eval.py
git commit -m "Add classification and detection metric helpers"
```

---

## Task 15: `eval/inference.py` — Oracle + Realistic 평가 경로

**Files:**
- Create: `src/disease_detection/eval/inference.py`
- Modify: `tests/test_eval.py` (append)

- [ ] **Step 1: 실패 테스트 작성**

```python
from pathlib import Path

from disease_detection.eval.inference import evaluate_classifier_oracle


def test_evaluate_classifier_oracle_returns_report(fixtures_dir, mocker):
    from disease_detection.data.classification_dataset import build_fireblight_crops

    items = build_fireblight_crops(fixtures_dir / "dummy_aihub")

    def fake_classifier_fn(batch):
        # Always predict positive with high probability
        import torch
        return torch.full((batch.shape[0],), 3.0)  # sigmoid(3)=0.95

    rep = evaluate_classifier_oracle(
        items,
        classifier_fn=fake_classifier_fn,
        batch_size=2,
    )
    assert rep.accuracy >= 0.0
    assert rep.recall == 1.0  # all-positive predictor, positives caught
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_eval.py -v` → ImportError.

- [ ] **Step 3: 구현**

`src/disease_detection/eval/inference.py`:
```python
"""End-to-end 평가 경로.

- `evaluate_classifier_oracle` — GT bbox로 crop한 분류 입력에 대한 분류기 단독 평가 (상한).
- `evaluate_pipeline_realistic` — Detector 예측 bbox를 활용한 2단계 파이프라인 평가.
"""
from __future__ import annotations

from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..data.classification_dataset import ClassificationCropDataset, CropItem
from ..data.transforms import build_classifier_eval_transform
from .metrics import ClassificationMetricsReport, compute_classification_report


def _collate_tensor_labels(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys


def evaluate_classifier_oracle(
    items: list[CropItem],
    classifier_fn: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int = 16,
    device: str = "cpu",
    threshold: float = 0.5,
) -> ClassificationMetricsReport:
    """GT bbox crop으로 분류기 단독 성능 측정."""
    ds = ClassificationCropDataset(items, transform=build_classifier_eval_transform())
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=_collate_tensor_labels)

    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for xs, ys in loader:
        xs = xs.to(device)
        logits = classifier_fn(xs)
        all_scores.append(torch.sigmoid(logits.detach().cpu()))
        all_labels.append(ys)

    y_score = torch.cat(all_scores)
    y_true = torch.cat(all_labels)
    return compute_classification_report(y_true, y_score, threshold=threshold)
```

(Realistic 평가는 `TwoStagePipeline`을 iterate하며 이미지 단위로 prob 집계; 동일 `compute_classification_report`로 리포팅. 구현은 Task 23에서 포함.)

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_eval.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/disease_detection/eval/inference.py tests/test_eval.py
git commit -m "Add Oracle evaluation path for classifier crops"
```

---

## Task 16: Hydra 설정 파일

**Files:**
- Create: `configs/config.yaml`
- Create: `configs/data/{aihub_pear,aihub_apple,aihub_combined}.yaml` (3개)
- Create: `configs/model/{detector_fasterrcnn,classifier_resnet18}.yaml` (2개)
- Create: `configs/trainer/{local_cpu,ubuntu_gpu}.yaml` (2개)
- Create: `configs/experiment/{fireblight_baseline,defect_baseline}.yaml` (2개)

- [ ] **Step 1: 루트 `configs/config.yaml`**

```yaml
defaults:
  - data: aihub_combined
  - model: classifier_resnet18
  - trainer: ubuntu_gpu
  - _self_

seed: 42
paths:
  dataset_root: ${oc.env:DATASET_ROOT,???}
  models_root: ${paths.dataset_root}/../models/disease_02

run:
  name: default
```

- [ ] **Step 2: 데이터 설정 3종**

`configs/data/aihub_pear.yaml`:
```yaml
name: aihub_pear
crops: [pear]
root: ${paths.dataset_root}/raw/aihub
splits_dir: ${paths.dataset_root}/splits
split: train
batch_size: 16
num_workers: 4
```

`configs/data/aihub_apple.yaml`:
```yaml
name: aihub_apple
crops: [apple]
root: ${paths.dataset_root}/raw/aihub
splits_dir: ${paths.dataset_root}/splits
split: train
batch_size: 16
num_workers: 4
```

`configs/data/aihub_combined.yaml`:
```yaml
name: aihub_combined
crops: [pear, apple]
root: ${paths.dataset_root}/raw/aihub
splits_dir: ${paths.dataset_root}/splits
split: train
batch_size: 16
num_workers: 4
```

- [ ] **Step 3: 모델 설정 2종**

`configs/model/detector_fasterrcnn.yaml`:
```yaml
_target_: disease_detection.models.detector.FasterRCNNModule
num_classes: 4
lr: 0.005
momentum: 0.9
weight_decay: 5e-4
lr_step: 10
lr_gamma: 0.1
```

`configs/model/classifier_resnet18.yaml`:
```yaml
_target_: disease_detection.models.classifier.PlantDefectClassifier
lr: 1e-4
weight_decay: 1e-2
pos_weight: null
t_max: 30
```

- [ ] **Step 4: Trainer 설정 2종**

`configs/trainer/ubuntu_gpu.yaml`:
```yaml
accelerator: gpu
devices: 1
precision: "16-mixed"
max_epochs: 50
gradient_clip_val: 1.0
log_every_n_steps: 20
early_stopping:
  monitor: val/loss
  patience: 5
  mode: min
checkpoint:
  monitor: val/loss
  mode: min
  save_top_k: 3
  filename: "{epoch}-{val/loss:.4f}"
```

`configs/trainer/local_cpu.yaml`:
```yaml
accelerator: cpu
devices: 1
precision: 32
max_epochs: 1
limit_train_batches: 10
limit_val_batches: 4
log_every_n_steps: 2
early_stopping: null
checkpoint:
  monitor: val/loss
  mode: min
  save_top_k: 1
  filename: "smoke-{epoch}"
```

- [ ] **Step 5: 실험 preset 2종**

`configs/experiment/fireblight_baseline.yaml`:
```yaml
# @package _global_
defaults:
  - override /data: aihub_combined
  - override /model: classifier_resnet18
  - override /trainer: ubuntu_gpu

run:
  name: fireblight_baseline

classifier:
  kind: fireblight
  label_source: aihub
  defect_threshold: null
  pos_weight: null
```

`configs/experiment/defect_baseline.yaml`:
```yaml
# @package _global_
defaults:
  - override /data: aihub_combined
  - override /model: classifier_resnet18
  - override /trainer: ubuntu_gpu

run:
  name: defect_baseline

classifier:
  kind: defect
  label_source: vlm
  vlm_jsonl: ${paths.dataset_root}/labels/vlm_severity/severity_labels.jsonl
  defect_threshold: 4
  pos_weight: null

trainer:
  max_epochs: 30
```

- [ ] **Step 6: Smoke test — Hydra가 config 로드하는지 확인**

```bash
python -c "
from hydra import compose, initialize_config_dir
from pathlib import Path
with initialize_config_dir(version_base='1.3', config_dir=str(Path('configs').resolve())):
    cfg = compose(config_name='config', overrides=['trainer=local_cpu'])
print(cfg.run.name)
print(cfg.trainer.accelerator)
"
```

Expected: `default` / `cpu` 출력.

- [ ] **Step 7: Commit**

```bash
git add configs/
git commit -m "Add Hydra config tree for data/model/trainer/experiment"
```

---

## Task 17: `scripts/download_aihub.sh` — 무결성 검증 스크립트

**Files:**
- Create: `scripts/download_aihub.sh` (실행 권한 포함)

- [ ] **Step 1: 스크립트 작성**

`scripts/download_aihub.sh`:
```bash
#!/usr/bin/env bash
# AIhub 배·사과 화상병 데이터셋 수동 다운로드 가이드 + 무결성 검증.
# 실제 다운로드는 AIhub 웹에서 로그인 후 수행해야 함.

set -euo pipefail

: "${DATASET_ROOT:?DATASET_ROOT 환경변수 필요}"
root="$DATASET_ROOT/raw/aihub"

echo "기대 경로: $root"
echo "필수 하위 구조:"
echo "  $root/pear/images/*.jpg"
echo "  $root/pear/annotations/*.{json,xml}"
echo "  $root/apple/images/*.jpg"
echo "  $root/apple/annotations/*.{json,xml}"
echo

ok=1
for crop in pear apple; do
  for kind in images annotations; do
    d="$root/$crop/$kind"
    if [[ ! -d "$d" ]]; then
      echo "❌ missing: $d"
      ok=0
      continue
    fi
    count=$(find "$d" -type f | wc -l | tr -d ' ')
    echo "✅ $d — $count 개 파일"
  done
done

if [[ "$ok" == 0 ]]; then
  echo
  echo "일부 경로 누락. AIhub에서 해당 데이터셋을 다운로드해 위 구조로 배치하세요."
  exit 1
fi

echo
echo "AIhub 디렉토리 구조 OK. 다음 단계:"
echo "  1) scripts/run_labeling.py 로 VLM 재라벨링 (macmini)"
echo "  2) scripts/sync_to_ubuntu.sh 로 Ubuntu 전송"
echo "  3) scripts/preprocess.py 로 split 생성 (Ubuntu)"
```

- [ ] **Step 2: 실행 권한 부여 + smoke test**

```bash
chmod +x scripts/download_aihub.sh
DATASET_ROOT=/tmp/empty_nonexistent scripts/download_aihub.sh || echo "Expected failure on empty root"
```

Expected: 누락 메시지 + exit 1. `echo` 한 마지막 문장 노출.

- [ ] **Step 3: Commit**

```bash
git add scripts/download_aihub.sh
git commit -m "Add AIhub directory integrity check script"
```

---

## Task 18: `scripts/run_labeling.py` — VLM 재라벨링 진입점

**Files:**
- Create: `scripts/run_labeling.py`

- [ ] **Step 1: 스크립트 작성**

```python
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
    per_bucket = max(1, sample_size // max(1, len(buckets)))
    selected: list[BatchJob] = []
    for key, jobs in buckets.items():
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
```

- [ ] **Step 2: Dry run으로 arg parser 동작 확인**

```bash
python scripts/run_labeling.py --help
```

Expected: usage 출력.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_labeling.py
git commit -m "Add run_labeling.py entry point with stratified sampling"
```

---

## Task 19: `scripts/sync_to_ubuntu.sh` — rsync 템플릿

**Files:**
- Create: `scripts/sync_to_ubuntu.sh`

- [ ] **Step 1: 스크립트 작성**

```bash
#!/usr/bin/env bash
# macmini → Ubuntu 데이터 동기화 (이미지·라벨).
# 필요한 환경변수: DATASET_ROOT, UBUNTU_USER, UBUNTU_HOST, REMOTE_DATASET_ROOT

set -euo pipefail

: "${DATASET_ROOT:?DATASET_ROOT 환경변수 필요}"
: "${UBUNTU_USER:?UBUNTU_USER 환경변수 필요}"
: "${UBUNTU_HOST:?UBUNTU_HOST 환경변수 필요}"
: "${REMOTE_DATASET_ROOT:?REMOTE_DATASET_ROOT 환경변수 필요}"

echo "Syncing $DATASET_ROOT → $UBUNTU_USER@$UBUNTU_HOST:$REMOTE_DATASET_ROOT"

rsync -avh --progress --partial \
  --exclude '__pycache__' --exclude '.DS_Store' \
  "$DATASET_ROOT/" \
  "$UBUNTU_USER@$UBUNTU_HOST:$REMOTE_DATASET_ROOT/"

echo "Sync complete."
```

- [ ] **Step 2: 실행 권한 + 간단 smoke test**

```bash
chmod +x scripts/sync_to_ubuntu.sh
# 환경변수 없이 실행 → 실패 메시지 확인
scripts/sync_to_ubuntu.sh || echo "expected failure without env vars"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/sync_to_ubuntu.sh
git commit -m "Add rsync template for macmini → Ubuntu dataset sync"
```

---

## Task 20: `scripts/preprocess.py` — split 생성

**Files:**
- Create: `scripts/preprocess.py`

- [ ] **Step 1: 스크립트 작성**

```python
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
```

- [ ] **Step 2: arg parser smoke test**

```bash
python scripts/preprocess.py --help
```

Expected: usage 출력.

- [ ] **Step 3: Commit**

```bash
git add scripts/preprocess.py
git commit -m "Add preprocess script for deterministic stratified splits"
```

---

## Task 21: `scripts/train_detector.py` — detector 학습 진입점

**Files:**
- Create: `scripts/train_detector.py`

- [ ] **Step 1: 스크립트 작성**

```python
#!/usr/bin/env python3
"""Faster R-CNN 학습 Hydra 진입점."""
from __future__ import annotations

import json
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from disease_detection.data.aihub import load_aihub_split
from disease_detection.data.detection_dataset import DetectionDataset
from disease_detection.data.transforms import (
    build_detector_eval_transform,
    build_detector_train_transform,
)
from disease_detection.models.detector import FasterRCNNModule
from disease_detection.utils.seeding import set_seed


def _detection_collate(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


def _build_loader(root: Path, indices: list[int], transform, batch_size: int, num_workers: int, shuffle: bool):
    entries = []
    for crop_name in ("pear", "apple"):
        entries.extend(load_aihub_split(root / crop_name))
    subset = [entries[i] for i in indices]
    ds = DetectionDataset(subset, transform=transform)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_detection_collate,
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    print(OmegaConf.to_yaml(cfg))

    aihub_root = Path(cfg.paths.dataset_root) / "raw" / "aihub"
    splits_path = Path(cfg.paths.dataset_root) / "splits" / "detector_split.json"
    split = json.loads(splits_path.read_text(encoding="utf-8"))

    train_loader = _build_loader(
        aihub_root,
        split["train"],
        build_detector_train_transform(),
        cfg.data.batch_size,
        cfg.data.num_workers,
        shuffle=True,
    )
    val_loader = _build_loader(
        aihub_root,
        split["val"],
        build_detector_eval_transform(),
        cfg.data.batch_size,
        cfg.data.num_workers,
        shuffle=False,
    )

    module = FasterRCNNModule(
        num_classes=cfg.model.num_classes,
        lr=cfg.model.lr,
        momentum=cfg.model.momentum,
        weight_decay=cfg.model.weight_decay,
        lr_step=cfg.model.lr_step,
        lr_gamma=cfg.model.lr_gamma,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{cfg.paths.models_root}/detector/{cfg.run.name}",
            monitor=cfg.trainer.checkpoint.monitor,
            mode=cfg.trainer.checkpoint.mode,
            save_top_k=cfg.trainer.checkpoint.save_top_k,
            filename=cfg.trainer.checkpoint.filename,
        )
    ]
    if cfg.trainer.get("early_stopping"):
        callbacks.append(
            EarlyStopping(
                monitor=cfg.trainer.early_stopping.monitor,
                patience=cfg.trainer.early_stopping.patience,
                mode=cfg.trainer.early_stopping.mode,
            )
        )

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", None),
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        limit_train_batches=cfg.trainer.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.trainer.get("limit_val_batches", 1.0),
        callbacks=callbacks,
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Hydra help 출력 확인**

```bash
python scripts/train_detector.py --help
```

Expected: Hydra help 출력.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_detector.py
git commit -m "Add train_detector.py Hydra entry point"
```

---

## Task 22: `scripts/train_classifier.py` — 화상병/범용 결함 학습 진입점

**Files:**
- Create: `scripts/train_classifier.py`

- [ ] **Step 1: 스크립트 작성**

```python
#!/usr/bin/env python3
"""PlantDefectClassifier 학습 Hydra 진입점. experiment preset에서 kind 분기."""
from __future__ import annotations

import json
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from disease_detection.data.classification_dataset import (
    ClassificationCropDataset,
    build_defect_crops,
    build_fireblight_crops,
)
from disease_detection.data.transforms import (
    build_classifier_eval_transform,
    build_classifier_train_transform,
)
from disease_detection.models.classifier import PlantDefectClassifier
from disease_detection.utils.seeding import set_seed


def _build_items(cfg: DictConfig):
    aihub_root = Path(cfg.paths.dataset_root) / "raw" / "aihub"
    kind = cfg.classifier.kind
    if kind == "fireblight":
        items = []
        for crop_name in ("pear", "apple"):
            items.extend(build_fireblight_crops(aihub_root / crop_name))
        split_name = "classifier_fireblight_split.json"
    elif kind == "defect":
        vlm_path = Path(cfg.classifier.vlm_jsonl)
        items = []
        for crop_name in ("pear", "apple"):
            items.extend(
                build_defect_crops(
                    aihub_root / crop_name,
                    vlm_path,
                    defect_threshold=cfg.classifier.defect_threshold,
                )
            )
        split_name = "classifier_defect_split.json"
    else:
        raise ValueError(f"Unknown classifier.kind: {kind}")
    return items, split_name


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    print(OmegaConf.to_yaml(cfg))

    items, split_name = _build_items(cfg)
    splits_path = Path(cfg.paths.dataset_root) / "splits" / split_name
    split = json.loads(splits_path.read_text(encoding="utf-8"))

    train_items = [items[i] for i in split["train"]]
    val_items = [items[i] for i in split["val"]]

    train_ds = ClassificationCropDataset(
        train_items, transform=build_classifier_train_transform()
    )
    val_ds = ClassificationCropDataset(
        val_items, transform=build_classifier_eval_transform()
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    module = PlantDefectClassifier(
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        pos_weight=cfg.classifier.get("pos_weight"),
        t_max=cfg.model.t_max,
    )

    ckpt_dir = f"{cfg.paths.models_root}/classifier_{cfg.classifier.kind}/{cfg.run.name}"
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor=cfg.trainer.checkpoint.monitor,
            mode=cfg.trainer.checkpoint.mode,
            save_top_k=cfg.trainer.checkpoint.save_top_k,
            filename=cfg.trainer.checkpoint.filename,
        )
    ]
    if cfg.trainer.get("early_stopping"):
        callbacks.append(
            EarlyStopping(
                monitor=cfg.trainer.early_stopping.monitor,
                patience=cfg.trainer.early_stopping.patience,
                mode=cfg.trainer.early_stopping.mode,
            )
        )

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        max_epochs=cfg.trainer.max_epochs,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        limit_train_batches=cfg.trainer.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.trainer.get("limit_val_batches", 1.0),
        callbacks=callbacks,
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: smoke test**

```bash
python scripts/train_classifier.py --help
```

Expected: Hydra help 출력.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_classifier.py
git commit -m "Add train_classifier.py with fireblight/defect kind dispatch"
```

---

## Task 23: `scripts/evaluate.py` — end-to-end 평가 진입점

**Files:**
- Create: `scripts/evaluate.py`

- [ ] **Step 1: 스크립트 작성**

```python
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
```

- [ ] **Step 2: Hydra help 출력 확인**

```bash
python scripts/evaluate.py --help
```

Expected: Hydra help 출력.

- [ ] **Step 3: Commit**

```bash
git add scripts/evaluate.py
git commit -m "Add evaluate.py Oracle evaluation entry point"
```

---

## Task 24: README에 사용법 섹션 추가

**Files:**
- Modify: `README.md` (append new `## 사용법` section)

- [ ] **Step 1: 섹션 작성**

README.md 하단 (`## 라이선스` 바로 위)에 추가:

```markdown
## 사용법

### 1. 환경 준비

```bash
# macmini
conda env create -f environment-macos.yml

# Ubuntu
conda env create -f environment-linux.yml

conda activate disease-detection
pip install -e .[dev]
cp .env.example .env    # DATASET_ROOT 등 편집
```

### 2. AIhub 다운로드 검증 (macmini)

```bash
export DATASET_ROOT=~/datasets/disease_02
scripts/download_aihub.sh    # 디렉토리 구조·개수 검증
```

### 3. VLM 재라벨링 (macmini, Max 20x)

```bash
python scripts/run_labeling.py --sample-size 3000 --model haiku \
  --out labels/vlm_severity/severity_labels.jsonl
```

중단/재개 자동. `.errors.jsonl`에 실패 기록. Claude Code CLI에 로그인 상태여야 함.

### 4. Ubuntu로 동기화

```bash
export UBUNTU_USER=... UBUNTU_HOST=... REMOTE_DATASET_ROOT=...
scripts/sync_to_ubuntu.sh
```

### 5. Split 생성 (Ubuntu)

```bash
python scripts/preprocess.py --defect-threshold 4
```

### 6. 학습 (Ubuntu)

```bash
# Detector
python scripts/train_detector.py trainer=ubuntu_gpu data=aihub_combined

# 화상병 분류기
python scripts/train_classifier.py +experiment=fireblight_baseline

# 범용 결함 분류기
python scripts/train_classifier.py +experiment=defect_baseline
```

### 7. 평가 (macmini 또는 Ubuntu)

```bash
python scripts/evaluate.py \
  +detector_ckpt=$MODELS/detector/best.ckpt \
  +fireblight_ckpt=$MODELS/classifier_fireblight/best.ckpt \
  +defect_ckpt=$MODELS/classifier_defect/best.ckpt
```

### 8. 테스트

```bash
pytest -q                    # 단위 테스트 (GPU·CLI 불필요)
pytest -m integration        # 실제 Claude CLI 사용 (수동)
pytest -m gpu                # CUDA 머신에서만
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Document usage pipeline in README"
```

---

## 최종 점검

- [ ] **전체 테스트 재실행**

```bash
pytest -q
```

Expected: 모든 단위 테스트 통과. 실패 시 해당 태스크로 돌아가 디버그.

- [ ] **Lint·포맷**

```bash
ruff check src tests scripts
black --check src tests scripts
```

Expected: 깨끗함. 경고 있으면 수정 후 새 커밋.

- [ ] **전체 커밋 확인**

```bash
git log --oneline | head -30
```

Task별로 1개 이상의 의미 있는 커밋이 있어야 함.

- [ ] **원격 push**

```bash
git push origin main
```

---

## 완료 기준 (Definition of Done)

- 모든 테스트 통과 (24 tasks × 커밋 ≥ 24)
- README 사용법 섹션이 실제로 실행 가능한 명령으로 구성됨
- macmini에서 `scripts/run_labeling.py --help` 실행 가능
- Ubuntu 데스크탑에서 Hydra 학습 스크립트가 (실제 데이터 없이도) config 로딩 단계까지 도달
- `original_code/`는 여전히 git 추적 밖

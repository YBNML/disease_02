# 배·사과 병충해 검출 (disease_02)

Phase 1 설계 문서 — 2026-04-21 초안, 2026-04-22 AIhub 실측 스키마 반영 업데이트

원본 PPT 파이프라인(D → A)을 재현하여 배(Pear)와 사과(Apple)의 **화상병(fireblight)** 및 **범용 결함(general defect)** 검출을 수행하는 프로젝트. PyTorch Lightning + Hydra 기반.

> **현재 상태**: Phase 1 구현 완료 + AIhub 실측 스키마 반영 완료. 학습 전 단계.

### 중요 업데이트 (v0.2, 2026-04-22)

AIhub 「과수화상병 촬영 이미지」 실제 스키마를 확인한 뒤 다음을 반영:

- **이미지당 bbox 1개** 고정. plant part (leaf/stem/fruit) 라벨이 없음 → Detector 는 **single-class "plant_roi"**.
- **화상병 이진화**는 `disease_code != 0` 로 단순화 (정상 vs 질병).
- **범용 결함은 4-부위 × 3-state multi-head 분류기**로 재설계. VLM v2 프롬프트가 한 bbox crop 에서 `leaf/branch/fruit/stem` 각각에 대해 `{defect, normal, absent}` 를 동시에 출력. 이전 단일 severity 스칼라(v1) 구조는 폐기.
- **2-stage 파이프라인 유지** (원본 `original_code` 가 택한 방향과 일치). 원거리 촬영 시 다중 plant 객체·배경 잡음 대응에 유리.

---

## 1. 목표 및 범위

`docs/20250620 Pear Plant Defect Detection.pptx`의 Pear Plant Defect Detection 파이프라인을 처음부터 재현하되, **사과까지 대상 작물 확장**. 성능 최적화와 원거리 이미지 강인성은 **Phase 2**에서 별도 설계 문서로 다룸.

### Phase 1 포함 범위

- AIhub에서 배·사과 화상병 데이터셋 수집
- **범용 결함 데이터셋**을 VLM 기반 심각도 재라벨링으로 생성 (샘플 10,000장 이하)
- 2단계 파이프라인:
  1. 식물 부위 검출 (Fast R-CNN, torchvision)
  2. 독립 이진 분류기 × 2 (ResNet18):
     - 화상병 분류기 (AIhub 원본 라벨 사용)
     - 범용 결함 분류기 (VLM 심각도 라벨 이진화)
- 학습: Ubuntu 데스크탑 (CUDA). 라벨링·개발·평가: macmini
- 평가는 **Oracle** (GT bbox 기준)과 **Realistic** (예측 bbox 기준) 두 설정으로 수행

### Phase 1 제외 범위

- `original_code/models/` 보조 모듈 (DeblurGANv2, SRNO, PseCo)
  - **PseCo는 영구 제외** (과일 수확량 분석용, 병충해와 무관)
  - DeblurGANv2·SRNO는 Phase 2에서 필요 시 재검토
- 원거리 이미지 성능 개선 (Phase 2)
- End-to-end detector 교체 (예: YOLO, DETR)
- 심각도 회귀 또는 다중 클래스 심각도 예측
- 작물 판별기 (입력 시점에 작물 정보 제공됨)

## 2. 주요 결정사항

| 항목 | 결정 |
|---|---|
| 대상 작물 | 배 + 사과 (통합 모델 1개) |
| 작물 판별 | 입력 메타데이터로 전달, 모델 내부 분기 없음 |
| 파이프라인 | 원본 D → A 구조 (2-stage) |
| Detector 클래스 | **single-class** (bg + plant_roi). AIhub 에 부위 라벨 없음 |
| 화상병 분류기 | ResNet18 binary. 라벨: AIhub `disease_code != 0` → 1 |
| 범용 결함 분류기 | ResNet18 **multi-head (4 부위 × 3 state)**. 라벨: VLM v2 |
| 범용 결함 부위 | `leaf / branch / fruit / stem` 각각 `{defect, normal, absent}` |
| VLM 라벨링 도구 | Claude Code CLI headless (`claude -p --model haiku`), macmini |
| VLM 프롬프트 | v2 (4부위 동시 라벨링, bbox crop 입력) |
| VLM 라벨링 규모 | 범용 결함용 3,000~5,000장 (화상병은 AIhub 원본 라벨 그대로) |
| 학습 호스트 | Ubuntu 데스크탑 (SSH 원격) |
| 개발·라벨링·평가 | macmini |
| 프레임워크 | PyTorch Lightning + Hydra |
| Detector backbone | Faster R-CNN ResNet50-FPN v2 |
| Classifier backbone | ResNet18 |
| 가상환경 | conda, OS별 분리 (`environment-macos.yml`, `environment-linux.yml`) |
| `original_code/` | `.gitignore`로 제외, 참고 전용 |

**심각도 이진화 조정 근거**: AIhub 과수원 이미지에서 severity 1~3의 경미한 외형적 결함은 거의 모든 이미지에 존재하므로 실질적 병충해 신호를 얻으려면 임계값을 높여야 함. 설정 파일로 언제든 변경 가능.

## 3. 아키텍처 및 데이터 흐름

### 전체 흐름

```
[macmini : 라벨링 / 개발 / 평가]
  AIhub 다운로드 (배·사과 화상병)
      ↓
  raw/aihub/{pear,apple}/{images, annotations}
      ↓ scripts/run_labeling.py (Claude Code CLI, Haiku)
  labels/vlm_severity/severity_labels.jsonl  (샘플 3~5k)
      ↓ rsync (scripts/sync_to_ubuntu.sh)
[Ubuntu 데스크탑 : 학습]
  scripts/preprocess.py
      ├─ Detector split (COCO format, GT bbox)
      └─ Classifier splits (GT crop)
            ├─ 화상병 분류기 (AIhub 원본 라벨 전체)
            └─ 범용 결함 분류기 (VLM 이진화 라벨 샘플)
      ↓
  scripts/train_detector.py            → models/detector/best.ckpt
  scripts/train_classifier.py (×2)     → models/classifier_fireblight/best.ckpt
                                         models/classifier_defect/best.ckpt
      ↓ rsync (best.ckpt만)
[macmini : 평가]
  scripts/evaluate.py  (Oracle + Realistic)
      → reports/{timestamp}/{metrics.json, confusion.png, samples/}
```

### 추론 파이프라인 (2단계)

```
이미지 + 작물 정보(메타데이터)
    ↓ Fast R-CNN
부위 bbox {leaf, stem, fruit}
    ↓ crop, 224×224 resize
    ├─ ResNet18 (화상병)   → P(fireblight)
    └─ ResNet18 (범용 결함) → P(defect)
```

Phase 1에서는 작물 정보가 메타데이터로만 전달됨 (두 분류기 모두 사용 안 함). 로깅·향후 조건부 모델링 여지 확보 목적으로 파이프라인에 관통.

## 4. 레포 구조

```
disease_02/
├── src/disease_detection/           # Python 패키지
│   ├── data/
│   │   ├── aihub.py                 # AIhub 파서 + 검증
│   │   ├── detection_dataset.py     # Fast R-CNN용 데이터셋
│   │   ├── classification_dataset.py# ResNet18용 데이터셋 (crop)
│   │   └── transforms.py            # torchvision v2 transforms
│   ├── models/
│   │   ├── detector.py              # Faster R-CNN LightningModule
│   │   ├── classifier.py            # PlantDefectClassifier (공용 클래스)
│   │   └── pipeline.py              # 2단계 추론 래퍼
│   ├── labeling/
│   │   ├── vlm_client.py            # Claude Code CLI subprocess 래퍼
│   │   ├── prompts.py               # PPT Slide 2 심각도 프롬프트 + 버전
│   │   └── batch_label.py           # 재개 가능한 배치 실행기
│   ├── eval/
│   │   ├── metrics.py               # torchmetrics 기반
│   │   └── inference.py             # Oracle + Realistic 평가 경로
│   └── utils/
│       ├── io.py                    # $DATASET_ROOT 경로 해석
│       └── seeding.py
├── configs/                         # Hydra 설정
│   ├── config.yaml                  # 루트
│   ├── data/{aihub_pear,aihub_apple,aihub_combined}.yaml
│   ├── model/{detector_fasterrcnn,classifier_resnet18}.yaml
│   ├── trainer/{local_cpu,ubuntu_gpu}.yaml
│   └── experiment/                  # 실험 프리셋
├── scripts/
│   ├── download_aihub.sh            # 다운로드 가이드 + 무결성 검사
│   ├── run_labeling.py              # VLM 재라벨링 (macmini)
│   ├── sync_to_ubuntu.sh            # rsync 템플릿
│   ├── preprocess.py                # crop + split
│   ├── train_detector.py
│   ├── train_classifier.py          # Hydra로 fireblight/defect 선택
│   └── evaluate.py
├── tests/
│   ├── fixtures/                    # 더미 이미지·어노테이션
│   ├── test_data.py
│   ├── test_labeling.py             # subprocess 모킹
│   ├── test_models.py               # forward shape + 10샘플 overfit sanity
│   └── test_pipeline.py             # 2단계 로직 (모킹)
├── docs/                            # (향후) 아키텍처 다이어그램, Phase 2 설계 등
├── environment-macos.yml            # macmini (MPS)
├── environment-linux.yml            # Ubuntu (CUDA)
├── environment-common.yml           # 공통 pin
├── pyproject.toml                   # 패키지 + ruff/black
├── .env.example                     # DATASET_ROOT 등
├── .gitignore                       # original_code/, data/, outputs/ 제외
└── README.md                        # 이 문서
```

## 5. 데이터 파이프라인

### 5.1 AIhub 다운로드 (macmini)

`scripts/download_aihub.sh`는 **자동 다운로더가 아닌 가이드 + 무결성 검증기**. AIhub은 인증 기반이라 다운로드는 수동. 다운로드 완료 후 체크섬·이미지 개수·어노테이션 무결성 확인.

예상 레이아웃:
```
$DATASET_ROOT/raw/aihub/
  pear/{images/*.jpg, annotations/*.{json,xml}}
  apple/{images/*.jpg, annotations/*.{json,xml}}
```

AIhub 어노테이션 포맷(JSON vs XML, 스키마)은 실제 다운로드 전까진 미정. 파서 모듈(`src/disease_detection/data/aihub.py`)은 포맷별 loader를 갖되 공통 dataclass로 통일.

### 5.2 VLM 재라벨링 (macmini)

- **도구**: Claude Code CLI headless (`claude -p "..." --model haiku`), `subprocess`로 호출
- **모델**: `claude-haiku-4-5` (심각도 평가에 충분, rate limit 소모 최소)
- **프롬프트**: PPT Slide 2 심각도 루브릭 원문 그대로. `prompts.py`에 `prompt_version` 포함 저장
- **샘플링**: `작물 × 부위 × 화상병 여부`로 stratified, 총 3,000~5,000장
- **출력**: `labels/vlm_severity/severity_labels.jsonl`, 이미지당 JSON 한 줄

```json
{
  "image_path": "raw/aihub/pear/images/xxx.jpg",
  "image_sha256": "...",
  "crop": "pear",
  "plant_part": "leaf",
  "classification": "DEFECT",
  "severity": 6,
  "explanation": "...",
  "model": "claude-haiku-4-5",
  "prompt_version": "v1",
  "timestamp": "2026-04-21T10:00:00Z"
}
```

- **재개**: 재시작 시 기존 `image_sha256` 스킵
- **재시도 & 백오프**: 이미지당 최대 3회, exponential backoff (60s → 최대 30분)
- **에러**: 영구 실패는 `errors.jsonl`에 보존하고 스킵, 라벨링은 계속

### 5.3 심각도 이진화

- `severity ≤ 3` → 정상(0)
- `severity ≥ 4` → 결함(1)
- 임계값은 config로 노출, 실험 가능

### 5.4 전처리·Split (Ubuntu)

`scripts/preprocess.py`:

- **Detector**: AIhub 어노테이션 → COCO 포맷 변환, `crop × part` stratified 80/10/10
- **Classifier**: GT bbox로 crop하여 `.png` 저장 (224×224 resize는 transform에서), 분류기별 별도 split
  - 화상병: AIhub 원본 라벨, 전체, `crop × label` stratified 80/10/10
  - 범용 결함: VLM 라벨 subset, `crop × part × label` stratified 80/10/10
- **결정성**: 고정 seed 기반, `splits/*.json`에 동결하여 git에 커밋

### 5.5 Augmentation (torchvision v2)

- **Detector**: horizontal flip, scale jitter (0.8~1.2), 가벼운 color jitter
- **Classifier**: flip, rotation ±15°, color jitter, 가벼운 CutOut

### 5.6 동기화 (macmini → Ubuntu)

`scripts/sync_to_ubuntu.sh`:
```bash
rsync -avh --progress "$LOCAL_DATASET_ROOT/" \
  "$UBUNTU_USER@$UBUNTU_HOST:$REMOTE_DATASET_ROOT/"
```

초기 1회 이미지 전체 전송, 이후는 라벨·신규 crop 증분만. NFS/sshfs는 선택 옵션.

## 6. 모델 및 학습

### 6.1 Detector

- `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2`, ImageNet pretrained
- 클래스: `background, leaf, stem, fruit` (4개, 작물 구분 없음)
- 입력 해상도: short side 800 (메모리 부족 시 640)
- Loss: torchvision 내장 (RPN + ROI)
- 체크포인트: val `mAP@0.5` 기준 top-3 저장

### 6.2 Classifier (ResNet18 × 2, 독립)

`PlantDefectClassifier` 클래스 하나를 **두 인스턴스**로 학습 (코드 재사용, 학습은 완전 독립).

- Backbone: `torchvision.models.resnet18(weights=IMAGENET1K_V1)`
- Head: 최종 FC → 1 logit, `BCEWithLogitsLoss`
- 입력: 224×224 crop, ImageNet mean/std 정규화
- **학습 시 crop**: GT bbox (detector 오류와 분리하여 분류기 성능 순수 측정)
- **추론 시 crop**: Detector 예측 bbox
- 클래스 불균형: 양성 비율 낮으면 `pos_weight` 또는 `WeightedRandomSampler`

**인스턴스 A — 화상병 분류기**
- 라벨: AIhub 원본 화상병 이진 라벨
- 데이터: AIhub crop 전체

**인스턴스 B — 범용 결함 분류기**
- 라벨: VLM `severity ≥ 4` → 1, `≤ 3` → 0
- 데이터: VLM 라벨링 subset (3~5k crop)
- 데이터 부족 → 강한 augmentation + `EarlyStopping` 필수

### 6.3 Hydra + Lightning 오케스트레이션

```bash
# Detector
python scripts/train_detector.py trainer=ubuntu_gpu data=aihub_combined

# 화상병 분류기
python scripts/train_classifier.py +experiment=fireblight_baseline

# 범용 결함 분류기
python scripts/train_classifier.py +experiment=defect_baseline
```

Trainer preset:
- `configs/trainer/ubuntu_gpu.yaml`: `accelerator=gpu, devices=1, precision=16-mixed, max_epochs=50(det)/30(cls)`, `EarlyStopping(patience=5)`, `ModelCheckpoint(top_k=3)`
- `configs/trainer/local_cpu.yaml`: macmini용 smoke test 프로필 (1 epoch, 10 batch)

Logger: TensorBoard(기본) + CSVLogger(백업). Seed: `seed_everything(42)`, 오버라이드 가능.

### 6.4 Optimizer & Scheduler

- **Detector**: SGD(lr=0.005, momentum=0.9, weight_decay=5e-4), StepLR(step=10, gamma=0.1) — torchvision 레퍼런스 레시피
- **Classifier**: AdamW(lr=1e-4, weight_decay=1e-2), CosineAnnealingLR

### 6.5 체크포인트

Ubuntu의 `$DATASET_ROOT/../models/disease_02/{detector, classifier_fireblight, classifier_defect}/`에 저장. 각 디렉토리에 top-3 + `best.ckpt` symlink. macmini로는 `best.ckpt`만 rsync.

## 7. 평가

### 7.1 메트릭

- **Detector**: `MeanAveragePrecision` (mAP@0.5, mAP@0.5:0.95, per-class AP)
- **Classifier**: Accuracy, Precision, Recall, **F1**, ROC-AUC, PR-AUC, confusion matrix
- **End-to-end**: 두 설정 동시 측정
  - **Oracle** — GT bbox → classifier (성능 상한)
  - **Realistic** — detector bbox → classifier (배포 근사)

### 7.2 정책: 양성 클래스 Recall 우선

병충해를 놓치는 비용이 오탐 비용보다 큰 도메인. F1과 `Recall@Precision≥0.7`을 동시 리포트하되, **모델 선택 기준은 Recall@Precision≥0.7**. F1은 동률 tie-break.

### 7.3 평가 스크립트

```bash
python scripts/evaluate.py \
  +detector_ckpt=models/detector/best.ckpt \
  +fireblight_ckpt=models/classifier_fireblight/best.ckpt \
  +defect_ckpt=models/classifier_defect/best.ckpt \
  data.split=test
```

출력: `reports/{timestamp}/{metrics.json, confusion.png, sample_predictions/}`. macmini(CPU/MPS) 및 Ubuntu(CUDA) 양쪽에서 실행 가능.

## 8. 테스트

- **단위 테스트 (pytest)**
  - `test_data.py`: AIhub 파서, JSONL 파서, split 결정성
  - `test_labeling.py`: VLM 호출 subprocess 모킹, 재개·재시도 경로
  - `test_models.py`: forward shape 검증, 10샘플 overfit sanity
  - `test_pipeline.py`: 2단계 로직 (detector + classifier 모킹)
- **마커**
  - `@pytest.mark.integration` — 실제 Claude CLI (수동만)
  - `@pytest.mark.gpu` — CUDA 필요 (macmini에서 skip)
- **Fixture**: `tests/fixtures/`에 소형 더미 이미지 ~20장
- **CI 없음** (Phase 1 범위 밖, 추후 도입 가능)

## 9. 재현성

- `seed_everything(42)` 기본, 오버라이드 가능
- Hydra `outputs/YYYY-MM-DD/HH-MM-SS/config.yaml`에 resolved config 자동 스냅샷
- VLM JSONL에 `prompt_version`·`model` 기록 → 라벨 생성 이력 추적
- Split을 `splits/*.json`으로 동결·git 커밋
- 완료된 실험의 resolved config는 `configs/experiment/`에 승격 (영속 프리셋화)

## 10. 에러 처리

- **VLM 호출 실패**: 재시도 + 백오프, 영구 실패는 `errors.jsonl`에 기록하고 라벨링 계속
- **VLM 응답 파싱 실패**: raw 응답을 `errors.jsonl`에 보존 (조용히 삭제하지 않음)
- **AIhub 무결성**: `scripts/download_aihub.sh`가 체크섬·개수 검증, 학습 진입 시 누락·손상 즉시 실패
- **학습 중단 복구**: Lightning resume-from-checkpoint, Hydra 출력 디렉토리 기준

## 11. 로깅 & 관측

- 학습: TensorBoard (loss, metrics, LR, epoch별 sample predictions)
- 라벨링: `tqdm` 진행바 + 주기적 `stats.json` (처리 수, 실패 수, 처리량)
- 추론: `logging` 모듈, INFO/DEBUG 레벨

## 12. Open Items (구현 중 해결)

- AIhub 어노테이션의 정확한 포맷·스키마 — 다운로드 후 확정. 파서는 pluggable로 설계
- AIhub 화상병 양성 비율 — `pos_weight` / sampler 선택 결정에 영향
- Max 20x 플랜에서 지속 라벨링 시 rate limit 실제 동작 — 재개 가능한 batch runner로 대응 예정
- **Phase 2 (원거리 강인성)** 설계 — Phase 1 baseline 완성 후 별도 문서로 작성

## 13. 환경 정보

- 라벨링·개발·평가: macmini (Apple Silicon, MPS)
- 학습: Ubuntu 데스크탑 (CUDA, SSH 원격 접속)
- Python 3.11, conda 기반
- Claude Code Max 20x 구독 활용 (API 별도 비용 없음)

---

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
python scripts/train_detector.py trainer=ubuntu_gpu data=aihub_combined model=detector_fasterrcnn

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

## 라이선스

현재 미정.

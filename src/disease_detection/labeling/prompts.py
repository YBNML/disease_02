"""VLM 재라벨링용 프롬프트.

PROMPT_VERSION 이력:
- v1: 단일 severity (0~10) + NORMAL/DEFECT. 이미지 전체 대상. **더 이상 사용 안 함** —
  AIhub 데이터 스키마 재확인 후 부위별 라벨링이 필요해져 v2로 대체.
- v2 (현재): 4-부위 (leaf/branch/fruit/stem) 각각에 대해
  `{present, defect, severity}` 동시 출력. VLM 입력은 bbox crop.

프롬프트 수정 시 `PROMPT_VERSION` 을 반드시 bump 하고, JSONL 라벨의 `prompt_version`
필드가 일치하는 샘플만 재사용할 것.
"""
from __future__ import annotations

PROMPT_VERSION: str = "v2"

SEVERITY_PROMPT_V2: str = (
    "You are inspecting a cropped region of an orchard image. Identify which plant "
    "parts are visible and assess whether each visible part shows any defect "
    "(fireblight, rot, necrosis, browning, holes, mold, pest damage, deformities, etc.).\n\n"
    "Consider exactly four plant parts:\n"
    "- `leaf`   : any leaf tissue\n"
    "- `branch` : woody branches/twigs/peduncle bases\n"
    "- `fruit`  : pears or apples (including immature fruit)\n"
    "- `stem`   : main stem/trunk region\n\n"
    "For each of the four parts, assign:\n"
    "- `state`: one of `defect`, `normal`, or `absent`.\n"
    "    - `absent`  — the part is NOT visible in the crop.\n"
    "    - `normal`  — the part is visible and appears healthy.\n"
    "    - `defect`  — the part is visible and shows a defect.\n"
    "- `severity`: integer 0–10. `0` when state is `normal` or `absent`; `1`–`10` when\n"
    "  state is `defect`, matching the rubric below.\n"
    "- `reason`: ≤ 10 words briefly citing the visible cue.\n\n"
    "Severity rubric (for `defect` state only):\n"
    "- **1–3**: minor or cosmetic (small dry spot, light browning)\n"
    "- **4–6**: moderate (noticeable discoloration/rot/damage on part of the tissue)\n"
    "- **7–10**: severe / widespread / critical damage\n\n"
    "### Return a single JSON object, no prose. Exactly these keys:\n"
    '```json\n{\n'
    '  "parts": {\n'
    '    "leaf":   {"state": "defect|normal|absent", "severity": 0-10, "reason": "..."},\n'
    '    "branch": {"state": "defect|normal|absent", "severity": 0-10, "reason": "..."},\n'
    '    "fruit":  {"state": "defect|normal|absent", "severity": 0-10, "reason": "..."},\n'
    '    "stem":   {"state": "defect|normal|absent", "severity": 0-10, "reason": "..."}\n'
    '  }\n'
    "}\n```\n\n"
    "### Examples:\n"
    "- Healthy pear leaf crop, no other parts visible:\n"
    '  {"parts": {"leaf": {"state":"normal","severity":0,"reason":"green intact"},\n'
    '             "branch":{"state":"absent","severity":0,"reason":""},\n'
    '             "fruit":{"state":"absent","severity":0,"reason":""},\n'
    '             "stem":{"state":"absent","severity":0,"reason":""}}}\n'
    "- Fireblight on leaves + healthy branch visible, no fruit/stem:\n"
    '  {"parts": {"leaf":{"state":"defect","severity":7,"reason":"necrotic brown tips"},\n'
    '             "branch":{"state":"normal","severity":0,"reason":"intact bark"},\n'
    '             "fruit":{"state":"absent","severity":0,"reason":""},\n'
    '             "stem":{"state":"absent","severity":0,"reason":""}}}\n'
)

# v1 은 더 이상 사용하지 않지만 이력 추적을 위해 상수만 남긴다 (raise NotImplementedError 방지).
SEVERITY_PROMPT_V1: str = SEVERITY_PROMPT_V2  # deprecated alias

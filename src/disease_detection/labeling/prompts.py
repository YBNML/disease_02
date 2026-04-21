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
    "### Return a single JSON object (no prose) with these fields:\n"
    "- `Classification`: `NORMAL` or `DEFECT`\n"
    "- `Severity`: `0` if NORMAL, `1-10` if DEFECT (match the rubric above)\n"
    "- `explanation`: brief (≤ 15 words), include defect type and plant part if applicable\n\n"
    '```json\n{"classification": "NORMAL|DEFECT", "severity": 0-10, "explanation": "..."}\n```\n\n'
    "### Examples:\n"
    '- {"classification": "NORMAL", "severity": 0, "explanation": "fruit is healthy"}\n'
    '- {"classification": "DEFECT", "severity": 2, "explanation": "minor leaf tip browning"}\n'
    '- {"classification": "DEFECT", "severity": 8, "explanation": "fruit partially rotten"}\n'
)

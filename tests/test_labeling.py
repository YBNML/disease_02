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

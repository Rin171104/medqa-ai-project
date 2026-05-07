from __future__ import annotations

import re
from typing import Optional, Tuple


def parse_mcq_output(text: str) -> Tuple[Optional[str], str]:
    """Parse model output into (answer_letter, explanation)."""

    cleaned = (text or "").strip()
    # Remove <think> blocks from explanation parsing
    cleaned_no_think = re.sub(r"(?is)<think>.*?</think>", "", cleaned).strip()

    # ==========================================
    # 1. EXTRACT ANSWER (PREFER TAGS)
    # ==========================================
    answer: Optional[str] = None

    tag_answer = re.search(r"(?is)<\s*answer\s*>\s*([ABCD])\s*<\s*/\s*answer\s*>", cleaned)
    if tag_answer:
        answer = tag_answer.group(1).upper()

    # Fallback: only accept answers that appear at line starts or after explicit keywords
    if not answer:
        fallback_patterns = [
            r"(?im)^\s*Answer\s*[:\-]\s*([ABCD])\b",
            r"(?im)^\s*Answer\s*(?:is)?\s*[:\-]?\s*([ABCD])\b",
            r"(?im)^\s*Correct\s*answer\s*[:\-]?\s*([ABCD])\b",
            r"(?im)^\s*Đáp\s*án\s*[:\-]\s*([ABCD])\b",
        ]
        for pat in fallback_patterns:
            m = re.search(pat, cleaned_no_think)
            if m:
                answer = m.group(1).upper()
                break

    # ==========================================
    # 2. EXTRACT EXPLANATION (PREFER TAGS)
    # ==========================================
    explanation = ""

    tag_explanation = re.search(
        r"(?is)<\s*explanation\s*>\s*(.+?)\s*<\s*/\s*explanation\s*>",
        cleaned,
    )
    if tag_explanation:
        explanation = tag_explanation.group(1).strip()

    if not explanation:
        exp_patterns = [
            r"(?ims)\bExplanation\s*[:\-]\s*(.+)$",
            r"(?ims)\bGiải\s*thích\s*[:\-]\s*(.+)$",
        ]

        for pat in exp_patterns:
            m = re.search(pat, cleaned)
            if m:
                explanation = m.group(1).strip()
                break

    # fallback explanation
    if not explanation:
        explanation = re.sub(
            r"(?im)^\s*(Answer|Correct answer|Đáp\s*án).*?$",
            "",
            cleaned_no_think,
        ).strip()

        if not explanation:
            explanation = cleaned_no_think

    return answer, explanation
from __future__ import annotations

import re
from typing import Optional, Tuple


def parse_mcq_output(text: str) -> Tuple[Optional[str], str]:
    """Parse model output into (answer_letter, explanation)."""

    cleaned = (text or "").strip()
    # Remove <think> blocks from explanation parsing
    cleaned_no_think = re.sub(r"(?is)<think>.*?</think>", "", cleaned).strip()

    # ==========================================
    # 1. EXTRACT ANSWER
    # ==========================================
    answer: Optional[str] = None

    answer_patterns = [
        r"(?im)^\s*Answer\s*[:\-]\s*([ABCD])\b",
        r"(?im)^\s*Answer\s*(?:is)?\s*[:\-]?\s*([ABCD])\b",
        r"(?im)^\s*Correct\s*answer\s*[:\-]?\s*([ABCD])\b",
        r"(?im)^\s*Đáp\s*án\s*[:\-]\s*([ABCD])\b",
        r"(?im)<\s*answer\s*>\s*([ABCD])\s*<\s*/\s*answer\s*>",
    ]

    for pat in answer_patterns:
        m = re.search(pat, cleaned_no_think)
        if m:
            # pick the letter group only
            letter = next((g for g in m.groups() if g and g.upper() in {"A", "B", "C", "D"}), None)
            if letter:
                answer = letter.upper()
            break

    # Fallback: only accept answers that appear at line starts or after explicit keywords
    if not answer:
        fallback_patterns = [
            r"(?im)^\s*([ABCD])\s*[\)\.:\-]",  # e.g. "B." or "C)"
            r"(?im)\b(the\s+answer\s+is|answer\s+is|correct\s+answer)\s*[:\-]?\s*([ABCD])\b",
        ]
        for pat in fallback_patterns:
            m = re.search(pat, cleaned)
            if m:
                letter = next((g for g in m.groups() if g and g.upper() in {"A", "B", "C", "D"}), None)
                if letter:
                    answer = letter.upper()
                    break

    # ==========================================
    # 2. EXTRACT EXPLANATION
    # ==========================================
    explanation = ""

    exp_patterns = [
        r"(?ims)\bExplanation\s*[:\-]\s*(.+)$",
        r"(?ims)\bGiải\s*thích\s*[:\-]\s*(.+)$",
        r"(?ims)<\s*explanation\s*>\s*(.+?)\s*<\s*/\s*explanation\s*>",
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
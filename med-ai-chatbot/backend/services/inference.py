from __future__ import annotations

from typing import Any, Dict

import torch
from fastapi import HTTPException

from backend.models.loader import load_model_and_tokenizer
from backend.utils.parser import parse_mcq_output


_bundle: Dict[str, Any] | None = None


def _get_bundle() -> Dict[str, Any]:
    global _bundle
    if _bundle is None:
        _bundle = load_model_and_tokenizer()
    return _bundle


def ask_mcq(question: str, options: Dict[str, str]) -> Dict[str, str]:
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is empty")

    for key in ("A", "B", "C", "D"):
        if key not in options or not str(options[key]).strip():
            raise HTTPException(status_code=400, detail=f"Missing option {key}")

    bundle = _get_bundle()
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]

    # 🔥 PROMPT MỀM HƠN (QUAN TRỌNG)
    user_text = (
        f"Question: {question.strip()}\n"
        "Options:\n"
        f"A. {options['A'].strip()}\n"
        f"B. {options['B'].strip()}\n"
        f"C. {options['C'].strip()}\n"
        f"D. {options['D'].strip()}\n\n"
        "Choose the correct answer (A/B/C/D) and explain briefly."
    )

    system_prompt = (
        "You are a medical AI assistant. "
        "Answer multiple choice medical questions correctly and explain briefly."
    )

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"System: {system_prompt}\n\nUser: {user_text}\n\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,   # 🔥 tăng lên
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # 🔥 PARSE
    answer, explanation = parse_mcq_output(raw_output)

    # 🔥 FALLBACK nếu parse fail
    if not answer:
        import re
        match = re.search(r"\b([A-D])\b", raw_output.upper())
        answer = match.group(1) if match else "Unknown"

    return {
        "answer": answer,
        "explanation": explanation or raw_output,
        "raw_output": raw_output,
    }
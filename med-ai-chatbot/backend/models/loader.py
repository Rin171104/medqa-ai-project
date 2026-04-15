from __future__ import annotations

import os
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


_model_bundle: Dict[str, Any] | None = None


def load_model_and_tokenizer() -> Dict[str, Any]:
    global _model_bundle

    if _model_bundle is not None:
        return _model_bundle

    base_model_name = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-3B")
    adapter_path = os.environ.get("ADAPTER_PATH", "./model/adapter")

    print("🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("🔄 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )

    print("🔄 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False   # 🔥 FIX CHÍNH
    )

    model.eval()

    print("✅ Model loaded successfully")

    _model_bundle = {
        "model": model,
        "tokenizer": tokenizer,
    }

    return _model_bundle
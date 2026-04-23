import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_model_bundle = None


def load_model_and_tokenizer():
    global _model_bundle

    if _model_bundle is not None:
        return _model_bundle

    base_model_name = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")
    adapter_path = os.environ.get("ADAPTER_PATH", "./model/adapter")
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    merged_model_config_path = os.path.join(adapter_path, "config.json")

    has_adapter = os.path.isfile(adapter_config_path)
    use_merged_model = (not has_adapter) and os.path.isfile(merged_model_config_path)

    tokenizer_source = adapter_path if use_merged_model else base_model_name

    if use_merged_model:
        print("⚠️ adapter_config.json not found. Using ADAPTER_PATH as a merged model directory.")

    print("🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    allow_cpu_offload = os.environ.get("ALLOW_CPU_OFFLOAD", "0").lower() in {"1", "true", "yes"}

    if use_cuda:
        # 4-bit quantization only supported on GPU.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=allow_cpu_offload,
        )

        device_map = "auto" if allow_cpu_offload else "cuda"

        print("🔄 Loading base model (4-bit)...")

        base_model = AutoModelForCausalLM.from_pretrained(
            adapter_path if use_merged_model else base_model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        raise RuntimeError("CUDA is not available. GPU-only loading is required.")

    if has_adapter:
        print("🔄 Loading LoRA adapter...")

        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=False,
        )
    else:
        model = base_model

    model.eval()

    print("✅ Model loaded (4-bit)")

    _model_bundle = {
        "model": model,
        "tokenizer": tokenizer,
    }

    return _model_bundle
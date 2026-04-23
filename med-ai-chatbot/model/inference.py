import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen3-8B"
adapter_path = "./model/adapter"

print("🔄 Loading model...")

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

# 🔥 FIX pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.float16 if torch.cuda.is_available() else torch.float32

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=dtype,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    trust_remote_code=True
)

# 🔥 FIX QUAN TRỌNG
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    is_trainable=False
)

model.eval()

print("✅ Model ready")


def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # 🔥 FIX decode (chỉ lấy phần output mới)
    generated = outputs[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(generated, skip_special_tokens=True)
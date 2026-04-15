import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-3B"
adapter_path = "./model/adapter"

print("🔄 Loading model...")

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

print("✅ Model ready")


def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
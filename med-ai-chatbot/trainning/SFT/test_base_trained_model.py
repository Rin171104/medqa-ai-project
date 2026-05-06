import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
# Mô hình gốc ban đầu
BASE_MODEL = "/data2/cmdir/home/ioit107/vnu/models/Qwen/Qwen3-8B" 
# Trọng số LoRA của mô hình SFT
LORA_PATH = "/data2/cmdir/home/ioit107/mqhuy/medModel/medqa-qwen-sft/final_model"

TEST_FILE_PATH = "/data2/cmdir/home/ioit107/mqhuy/medModel/MedQA-USMLE/questions/US/test.jsonl" 
OUTPUT_RESULT_PATH = "medqa_sft_test_results.jsonl"

# ==========================================
# 2. TẢI MODEL VÀ TOKENIZER LÊN GPU
# ==========================================
print("Đang tải Tokenizer từ mô hình gốc...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print("Đang tải Base Model lên GPU...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)

print("Đang ghép trọng số LoRA SFT vào mô hình...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# ==========================================
# 3. CHUẨN BỊ DỮ LIỆU TEST
# ==========================================
# Prompt SFT: Yêu cầu trả lời trực tiếp
PROMPT_TEMPLATE = """A conversation between User and Assistant.

User:
{question}

Options:
{options}

Assistant:
"""

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

print(f"\nĐang Test mô hình SFT {LORA_PATH}...")
print(f"Đang tải dữ liệu từ {TEST_FILE_PATH}...")
test_data = load_jsonl(TEST_FILE_PATH)
print(f"Tổng số câu hỏi cần test: {len(test_data)}\n")

# Các biến đếm kết quả
correct_count = 0
results = []

# ==========================================
# 4. CHẠY VÒNG LẶP ĐÁNH GIÁ (EVALUATION)
# ==========================================
for item in tqdm(test_data, desc="Đang đánh giá mô hình SFT"):
    question = item["question"]
    options_dict = item.get("options", {})
    options_text = "\n".join([f"{k}. {v}" for k, v in options_dict.items()])
    
    gold_answer = item.get("answer_idx", item.get("answer", "")).strip().upper()

    prompt = PROMPT_TEMPLATE.format(question=question, options=options_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64, # Rút ngắn lại vì SFT chỉ sinh đáp án, không sinh suy luận
            temperature=0.1,    
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # --- KIỂM TRA ĐÁP ÁN <answer> ---
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    if answer_match:
        model_answer = answer_match.group(1).strip().upper()
    else:
        # Trong trường hợp SFT quên sinh thẻ <answer>, lấy thẳng ký tự đầu tiên
        model_answer = response.strip()[0:1].upper() if response.strip() else "LỖI"

    is_correct = (model_answer == gold_answer)
    if is_correct:
        correct_count += 1

    # Lưu log chi tiết
    results.append({
        "question": question,
        "gold_answer": gold_answer,
        "model_answer": model_answer,
        "is_correct": is_correct,
        "full_response": response
    })

# ==========================================
# 5. TỔNG KẾT KẾT QUẢ
# ==========================================
accuracy = (correct_count / len(test_data)) * 100
print("\n" + "="*55)
print("BÁO CÁO KẾT QUẢ KIỂM THỬ SFT (TEST REPORT)")
print("="*55)
print(f"Tổng số câu hỏi    : {len(test_data)}")
print(f"Số câu trả lời đúng: {correct_count}")
print(f"Độ chính xác (Acc) : {accuracy:.2f}%")
print("="*55)

with open(OUTPUT_RESULT_PATH, "w", encoding="utf-8") as f:
    for res in results:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")

print(f"Chi tiết từng câu hỏi đã được lưu tại: {OUTPUT_RESULT_PATH}")
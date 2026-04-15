# Med AI Chatbot (MedQA)

Chatbot hỗ trợ trả lời **câu hỏi trắc nghiệm y khoa (A/B/C/D)** dựa trên mô hình ngôn ngữ + LoRA adapter.

> Lưu ý: Dự án phục vụ học tập/nghiên cứu. Không thay thế tư vấn chẩn đoán/điều trị từ nhân viên y tế.

## Tính năng

- Nhập câu hỏi + 4 lựa chọn trên giao diện Streamlit
- Backend FastAPI cung cấp API `/chat/ask`
- Inference bằng Transformers + PEFT (LoRA adapter)
- Tự parse đầu ra để lấy `answer` và `explanation`

## Công nghệ

- **Frontend**: Streamlit
- **Backend**: FastAPI + Uvicorn
- **AI/Model**: PyTorch, Transformers, PEFT (LoRA), Accelerate

## Cấu trúc thư mục

- `backend/`: FastAPI app (routes, services, model loader)
- `frontend/`: Streamlit UI
- `model/adapter/`: LoRA adapter (weights + config)
- `data/`: dữ liệu MedQA đã làm sạch (jsonl) và sách tham khảo
- `notebooks/`: notebook train/test

## Yêu cầu

- Python 3.10+ (khuyến nghị)
- (Tuỳ chọn) GPU CUDA để chạy nhanh hơn

## Cài đặt

Tại thư mục gốc dự án (nơi có `requirements.txt`):

```bash
python -m venv venv
# Windows
venv\\Scripts\\activate

pip install -r requirements.txt
```

## Cấu hình model

Backend sẽ load **base model** và gắn **LoRA adapter**.

Biến môi trường (tuỳ chọn):

- `BASE_MODEL`: tên model trên Hugging Face (mặc định: `Qwen/Qwen2.5-3B`)
- `ADAPTER_PATH`: đường dẫn tới adapter (mặc định: `./model/adapter`)

Ví dụ (PowerShell):

```powershell
$env:BASE_MODEL = "Qwen/Qwen2.5-3B"
$env:ADAPTER_PATH = "./model/adapter"
```

> Lần chạy đầu tiên có thể mất thời gian để tải base model từ Hugging Face.

## Chạy backend (FastAPI)

Tại thư mục gốc dự án:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Kiểm tra:

- Health check: `GET http://127.0.0.1:8000/health`
- API chính: `POST http://127.0.0.1:8000/chat/ask`

### Ví dụ gọi API

```bash
curl -X POST "http://127.0.0.1:8000/chat/ask" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"...\",\"options\":{\"A\":\"...\",\"B\":\"...\",\"C\":\"...\",\"D\":\"...\"}}"
```

Response:

```json
{
  "answer": "A",
  "explanation": "...",
  "raw_output": "..."
}
```

## Chạy frontend (Streamlit)

Mở terminal khác, tại thư mục gốc dự án:

```bash
streamlit run frontend/streamlit_app.py
```

Frontend mặc định gọi backend tại `http://127.0.0.1:8000/chat/ask`.

## Ghi chú triển khai

- Model được cache trong process backend (load một lần khi gọi lần đầu).
- Khi deploy production, cân nhắc chạy Uvicorn/Gunicorn và tắt `--reload`.

## License

Chưa khai báo.

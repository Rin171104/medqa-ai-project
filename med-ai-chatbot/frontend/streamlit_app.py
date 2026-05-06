import streamlit as st
import requests
import re

# ==============================
# CONFIG
# ==============================
API_URL = "http://127.0.0.1:8000/chat/ask"

st.set_page_config(page_title="Medical AI Assistant", layout="centered")

# ==============================
# HEADER
# ==============================
st.markdown("# 🩺 Medical AI Assistant")
st.markdown("---")

# ==============================
# SESSION STATE
# ==============================
if "history" not in st.session_state:
    st.session_state.history = []
if "question" not in st.session_state:
    st.session_state.question = ""
for key in ("opt_a", "opt_b", "opt_c", "opt_d"):
    if key not in st.session_state:
        st.session_state[key] = ""

# ==============================
# INPUT
# ==============================
st.markdown("### Nhập câu hỏi")
question = st.text_area(
    "",
    placeholder="Nhập câu hỏi y khoa...",
    height=100,
    key="question",
)

st.markdown("### Options")
col1, col2 = st.columns(2)

with col1:
    opt_a = st.text_input("A", key="opt_a")
    opt_c = st.text_input("C", key="opt_c")

with col2:
    opt_b = st.text_input("B", key="opt_b")
    opt_d = st.text_input("D", key="opt_d")

options = {
    "A": opt_a,
    "B": opt_b,
    "C": opt_c,
    "D": opt_d
}

def _clear_inputs():
    st.session_state.question = ""
    st.session_state.opt_a = ""
    st.session_state.opt_b = ""
    st.session_state.opt_c = ""
    st.session_state.opt_d = ""


def _delete_history_item(index: int) -> None:
    if 0 <= index < len(st.session_state.history):
        st.session_state.history.pop(index)


def _extract_think(raw_output: str | None) -> str:
    if not raw_output:
        return ""
    match = re.search(r"(?is)<think>\s*(.*?)\s*</think>", raw_output)
    return match.group(1).strip() if match else ""


# ==============================
# BUTTON
# ==============================
col_btn1, col_btn2 = st.columns([3, 1])

with col_btn1:
    ask = st.button("Ask AI", type="primary", use_container_width=True)

with col_btn2:
    clear = st.button("Clear", use_container_width=True, on_click=_clear_inputs)

# ==============================
# CLEAR
# ==============================
if clear:
    st.rerun()

# ==============================
# CALL API
# ==============================
if ask:
    if not question.strip():
        st.error("Vui lòng nhập câu hỏi")
    elif not all(options[k].strip() for k in ("A", "B", "C", "D")):
        st.error("Vui lòng nhập đầy đủ 4 lựa chọn")
    else:
        with st.spinner("Đang hỏi AI..."):
            try:
                res = requests.post(API_URL, json={
                    "question": question,
                    "options": options
                })

                data = res.json()

                st.session_state.history.append({
                    "question": question,
                    "options": options,
                    "answer": data.get("answer"),
                    "explanation": data.get("explanation"),
                    "raw_output": data.get("raw_output"),
                })

            except Exception as e:
                st.error(f"Lỗi kết nối backend: {e}")

# ==============================
# RESULT
# ==============================
st.markdown("---")
st.markdown("### 📄 Kết quả")

if st.session_state.history:
    last = st.session_state.history[-1]

    st.success(f"→ Answer: {last['answer']}")
    st.text_area("Explanation", value=last.get("explanation") or "", height=200, disabled=True)
    st.text_area("Think", value=_extract_think(last.get("raw_output")) or "", height=260, disabled=True)
else:
    st.info("Chưa có kết quả")

# ==============================
# HISTORY
# ==============================
st.markdown("---")
st.markdown("### 🧾 Lịch sử chat")

if not st.session_state.history:
    st.caption("(Trống)")
else:
    for i, item in enumerate(reversed(st.session_state.history), start=1):
        original_index = len(st.session_state.history) - i
        col_hist_title, col_hist_delete = st.columns([6, 1])
        with col_hist_title:
            st.markdown(f"**Câu {original_index + 1}:** {item['question']}")
        with col_hist_delete:
            st.button("Xóa", key=f"delete_{original_index}", on_click=_delete_history_item, args=(original_index,))

        with st.expander("Xem chi tiết"):
            st.write("**Options:**")
            st.write(f"A. {item['options']['A']}")
            st.write(f"B. {item['options']['B']}")
            st.write(f"C. {item['options']['C']}")
            st.write(f"D. {item['options']['D']}")

            st.write("**Kết quả:**")
            st.write(f"Answer: {item['answer']}")
            st.text_area("Explanation", value=item.get("explanation") or "", height=160, disabled=True)
            st.text_area("Think", value=_extract_think(item.get("raw_output")) or "", height=200, disabled=True)
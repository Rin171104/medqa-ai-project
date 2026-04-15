import streamlit as st
import requests

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

# ==============================
# INPUT
# ==============================
st.markdown("### Nhập câu hỏi")
question = st.text_area(
    "",
    placeholder="Nhập câu hỏi y khoa...",
    height=100
)

st.markdown("### Options")
col1, col2 = st.columns(2)

with col1:
    opt_a = st.text_input("A")
    opt_c = st.text_input("C")

with col2:
    opt_b = st.text_input("B")
    opt_d = st.text_input("D")

options = {
    "A": opt_a,
    "B": opt_b,
    "C": opt_c,
    "D": opt_d
}

# ==============================
# BUTTON
# ==============================
col_btn1, col_btn2 = st.columns([3, 1])

with col_btn1:
    ask = st.button("Ask AI", type="primary", use_container_width=True)

with col_btn2:
    clear = st.button("Clear", use_container_width=True)

# ==============================
# CLEAR
# ==============================
if clear:
    st.session_state.history = []
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
                    "explanation": data.get("explanation")
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
    st.write(f"→ Explanation: {last['explanation']}")
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
        with st.expander(f"Câu {len(st.session_state.history)-i+1}: {item['question'][:50]}..."):
            st.write("**Options:**")
            st.write(f"A. {item['options']['A']}")
            st.write(f"B. {item['options']['B']}")
            st.write(f"C. {item['options']['C']}")
            st.write(f"D. {item['options']['D']}")

            st.write("**Kết quả:**")
            st.write(f"Answer: {item['answer']}")
            st.write(f"Explanation: {item['explanation']}")
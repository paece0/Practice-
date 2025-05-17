import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="個性化聊天 AI", page_icon="🧠")
st.title("🧠 個性化聊天 AI")

# 載入模型（只執行一次）
@st.cache_resource
def load_model():
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# 初始化狀態
if "persona" not in st.session_state:
    st.session_state.persona = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# 個性輸入
if not st.session_state.persona:
    st.session_state.persona = st.text_input("請先輸入 AI 的個性（例如：風趣幽默、冷靜分析）")
    if not st.session_state.persona:
        st.stop()
    st.success("✅ 個性設定成功！開始對話吧")

# 顯示歷史對話（有頭像）
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# 使用 chat_input 輸入
user_input = st.chat_input("你想說什麼？")
if user_input:
    # 顯示使用者訊息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # 製作完整 prompt，加強指令避免跑題
    full_prompt = (
        f"你是一個{st.session_state.persona}風格的 AI，請針對使用者問題直接回應，避免離題或廢話。\n\n"
        f"使用者問：「{user_input}」\n"
        f"AI 回答："
    )
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt")

    # 模型生成
    output_ids = model.generate(
        input_ids,
        max_length=256,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7
    )
    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    # 顯示 AI 回應
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)

# 重新開始按鈕
if st.button("🔄 重新開始對話"):
    st.session_state.messages = []
    st.session_state.persona = ""
    st.experimental_rerun()

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="個性化聊天 AI", page_icon="🤖")
st.title("🧠 個性化聊天 AI")

# 模型載入
@st.cache_resource
def load_model():
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# 初始化 session_state
if "persona" not in st.session_state:
    st.session_state.persona = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# 個性輸入
if not st.session_state.persona:
    st.session_state.persona = st.text_input("請先輸入 AI 的個性描述，例如「風趣幽默」、「冷靜分析」")
    if not st.session_state.persona:
        st.stop()
    st.success("✅ 個性設定成功！請開始對話")

# 顯示歷史訊息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 使用 chat_input 輸入
user_input = st.chat_input("你想說什麼？")
if user_input:
    # 顯示使用者訊息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 準備 AI 輸入
    full_input = f"{st.session_state.persona}風格的 AI 回應：「{user_input}」"
    input_ids = tokenizer.encode(full_input + tokenizer.eos_token, return_tensors="pt")

    # 模型生成
    output_ids = model.generate(
        input_ids,
        max_length=256,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.75
    )
    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    # 顯示 AI 回應
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# 重新開始按鈕
if st.button("🔄 重新開始對話"):
    st.session_state.messages = []
    st.session_state.persona = ""
    st.experimental_rerun()

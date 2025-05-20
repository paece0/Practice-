import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="個性化聊天 AI", page_icon="🧠")
st.title("🧠 個性化聊天 AI（輕量穩定版）")

@st.cache_resource
def load_model():
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

if "persona" not in st.session_state:
    st.session_state.persona = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.persona:
    st.session_state.persona = st.text_input("請輸入 AI 個性（例如：幽默風趣、冷靜理性）")
    if not st.session_state.persona:
        st.stop()
    st.success("✅ 已設定個性！請開始對話")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

user_input = st.chat_input("你想說什麼？")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    prompt = (
        f"你是一個{st.session_state.persona}風格的 AI。請自然簡潔地回答使用者。\n"
        f"使用者說：「{user_input}」\n"
        f"AI 回應："
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(
        input_ids,
        max_length=200,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)

if st.button("🔄 重新開始"):
    st.session_state.messages = []
    st.session_state.persona = ""
    st.experimental_rerun()

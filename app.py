import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="個性化聊天 AI", page_icon="🧠")
st.title("🧠 個性化聊天 AI")

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
    st.session_state.persona = st.text_input("請先輸入 AI 的個性（例如：風趣幽默、冷靜分析）")
    if not st.session_state.persona:
        st.stop()
    st.success("✅ 個性設定成功！開始對話吧")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

user_input = st.chat_input("你想說什麼？")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    full_prompt = (
        f"你是一個{st.session_state.persona}風格的聊天 AI。請用自然、親切的方式回應使用者，"
        f"回答要簡單、貼近話題，不要加入無關的資訊或虛構故事。\n\n"
        f"使用者說：「{user_input}」\n"
        f"AI 回應："
    )
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt")

    output_ids = model.generate(
        input_ids,
        max_length=150,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7
    )
    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)

if st.button("🔄 重新開始對話"):
    st.session_state.messages = []
    st.session_state.persona = ""
    st.experimental_rerun()

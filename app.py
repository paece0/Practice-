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

# 初始化狀態
if "persona" not in st.session_state:
    st.session_state.persona = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# 設定 AI 個性
if not st.session_state.persona:
    st.session_state.persona = st.text_input("請先輸入 AI 的個性（例如：風趣幽默、冷靜分析）")
    if not st.session_state.persona:
        st.stop()
    st.success("✅ 個性設定成功！開始對話吧")

# 顯示對話歷史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# 使用者輸入
user_input = st.chat_input("你想說什麼？")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # 強化 Prompt，限制回覆只與對話有關，且避免虛構
    full_prompt = (
        f"你是一個風格為「{st.session_state.persona}」的聊天 AI。"
        "請用自然、親切且簡潔的語言回答使用者的問題。"
        "不要提供未經證實的資訊或故事，只根據使用者的輸入回答。"
        f"\n\n使用者說：「{user_input}」\nAI 回應："
    )

    input_ids = tokenizer.encode(full_prompt, return_tensors="pt")

    output_ids = model.generate(
        input_ids,
        max_length=150,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=40,
        top_p=0.85,
        temperature=0.7,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True).strip()

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)

# 重新開始按鈕
if st.button("🔄 重新開始對話"):
    st.session_state.messages = []
    st.session_state.persona = ""
    st.experimental_rerun()

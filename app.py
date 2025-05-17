import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.title("🧠 個性化聊天 AI（輕量穩定版）")

# 載入模型（distilgpt2 較輕量）
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 初始化狀態
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "persona_set" not in st.session_state:
    st.session_state.persona_set = False
if "persona" not in st.session_state:
    st.session_state.persona = ""

# 使用者輸入個性
if not st.session_state.persona_set:
    st.session_state.persona = st.text_input("請輸入 AI 的個性描述（例如：風趣幽默、冷靜分析）")
    if st.session_state.persona:
        st.session_state.persona_set = True
        st.write("✅ 個性設定成功！")
    st.stop()

# 使用者輸入訊息
user_input = st.text_input("你說：")

if st.button("送出") and user_input:
    # 組合輸入文本（加上個性）
    full_input = f"{st.session_state.persona}風格的 AI 回應：「{user_input}」"
    input_ids = tokenizer.encode(full_input + tokenizer.eos_token, return_tensors='pt')

    # 包含歷史
    if st.session_state.chat_history_ids is not None:
        input_ids = torch.cat([st.session_state.chat_history_ids, input_ids], dim=-1)

    # 生成回應（使用隨機參數避免重複、增強人味）
    output_ids = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.75
    )

    # 解碼與顯示
    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    st.write("💬 AI：" + response)

    # 更新歷史
    st.session_state.chat_history_ids = output_ids

# Reset
if st.button("🔄 重新開始"):
    st.session_state.chat_history_ids = None
    st.session_state.persona_set = False
    st.session_state.persona = ""
    st.experimental_rerun()

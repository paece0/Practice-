from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st

# 載入模型
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

st.title("自訂個性聊天 AI")

# 使用者輸入 AI 個性描述
persona = st.text_input("請輸入 AI 個性描述（例如：風趣幽默）")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

user_input = st.text_input("你說：")

if st.button("送出") and user_input:
    # 將個性加入對話歷史起始
    if st.session_state.chat_history is None:
        new_user_input_ids = tokenizer.encode(persona + " " + user_input + tokenizer.eos_token, return_tensors='pt')
    else:
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    if st.session_state.chat_history is None:
        st.session_state.chat_history = new_user_input_ids
    else:
        st.session_state.chat_history = torch.cat([st.session_state.chat_history, new_user_input_ids], dim=-1)

    output_ids = model.generate(
        st.session_state.chat_history,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    response = tokenizer.decode(output_ids[:, st.session_state.chat_history.shape[-1]:][0], skip_special_tokens=True)

    st.session_state.chat_history = output_ids

    st.write("AI：" + response)

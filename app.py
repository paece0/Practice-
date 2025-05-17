import streamlit as st

st.title("自訂個性聊天 AI")

# 使用者輸入 AI 個性描述
persona = st.text_input("請輸入 AI 個性描述（例如：風趣幽默）")

# 使用者輸入訊息
user_input = st.text_input("你說：")

if st.button("送出") and user_input:
    # 模擬回應邏輯
    response = f"（{persona} 的 AI）回應：「{user_input}，真有趣呢！」"
    st.write("AI：" + response)

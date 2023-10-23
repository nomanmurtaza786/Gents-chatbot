import os

import openai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from supa_bot import get_answer

load_dotenv()

st.title("GEN-TS HR Partner")

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
    
}

openai.api_key = os.getenv("OPENAI_API_KEY", "default_key")
uploaded_file = st.file_uploader(
    "Upload a Data file",
    type=["csv", "pdf"],
    help="Various File formats are Support",
    # on_change=clear_submit,
)

if uploaded_file is not None:
    print(f"==>> uploaded_file: {uploaded_file}")
 
    

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
        
            message_placeholder.markdown()
    st.session_state.messages.append({"role": "assistant", "content": full_response})
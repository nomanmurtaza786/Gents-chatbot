import os

import openai
import streamlit as st
from dotenv import load_dotenv
from key_store import (OPENAI_API_KEY,USER_ID)
from chains_bot import get_answer
from gen_embeddings import generateEmbeddings
from supabase_db import getSimilarDocuments
from PyPDF2 import PdfFileReader, PdfFileWriter,PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()

def getChatUi(user:str):
            openai.api_key = OPENAI_API_KEY
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
                    response = get_answer(prompt,user)
                    st.markdown(response["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                    

st.title("GEN-TS HR Partner")

user = st.text_input("Enter user id", value="dde6fc19-73fe-4ea6-a98b-a11485c710df",help="Pass the userId of which you are storing the document for. Other user's wont be able to access this doc then")

values = ['Yes','No']
option = st.selectbox(
    'Want to chat with previous docs',
    (values ),index=0)


if option == 'No' and user !=None:
    print(user)
    pdf = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=False)
    if pdf is not None:
            generateEmbeddings(pdf,user)
            getChatUi(user)
else:
    getChatUi(user)            




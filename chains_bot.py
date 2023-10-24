import os

from dotenv import load_dotenv
from langchain import HuggingFaceHub
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chains import (APIChain, ConversationalRetrievalChain,
                              RetrievalQA)
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate

from supabase_db import get_vector_store_retriever

load_dotenv()

openApiKey: str = os.getenv("OPENAI_API_KEY", "default_key")
huggingFaceApikey = os.getenv("API_KEY", "default_key")
openAillm = OpenAI(
    model="text-davinci-003",
    openai_api_key=openApiKey,
)
template = """ 
    Question: {question} 
    think step by step
    Answer: 
    """
prompt = PromptTemplate(template=template, input_variables=["question"])

chatLLM = ChatOpenAI(temperature=0.1,)
llm_chain = LLMChain(llm=chatLLM, prompt=prompt,verbose=True,)
#print("predict", chatLLM.predict('Captial of USA'))

crc = ConversationalRetrievalChain.from_llm(llm=chatLLM, retriever=get_vector_store_retriever(), verbose=True,)

api_chain = APIChain.from_llm_and_api_docs(llm=chatLLM, api_docs='' ,verbose=True,)


def get_answer(question: str, chat_history: list):
    result = crc({"question": question, "chat_history": chat_history})
    return result["answer"]

def callingApiChain(question: str, chat_history: list):
    result = crc({"question": question, "chat_history": chat_history})
    return result["answer"]



     
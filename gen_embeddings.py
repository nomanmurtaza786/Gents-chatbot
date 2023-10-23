
import os

from dotenv import load_dotenv

from langchain import HuggingFaceHub
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from supabase_db import getSimilarDocuments

embeddingOpenAi= OpenAIEmbeddings()

loader = PyPDFLoader('/Users/nomanmurtaza/Documents/Noman_Murtaza_CV.pdf')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
pages = loader.load_and_split(text_splitter=text_splitter)


# for page in element:
#     textLine:str = page.page_content.replace("\n", " ")
#     meta = page.metadata
#     embed=embeddingOpenAi.embed_query(textLine)
#     saveToSupabase(textLine, meta, embed)
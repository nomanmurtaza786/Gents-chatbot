import os

from dotenv import load_dotenv
from langchain.agents import (AgentExecutor, AgentType, Tool, create_sql_agent,
                              initialize_agent, load_tools)
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chains import APIChain, ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.sql_database import SQLDatabase
from key_store import (OPENAI_API_KEY, DB_CONNECTION_STR)
from supabase_db import get_vector_store_retriever

load_dotenv()

openApiKey: str = OPENAI_API_KEY
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

def getcrc(other,user:str =None):
    return( ConversationalRetrievalChain.from_llm(llm=chatLLM, retriever=get_vector_store_retriever(user), verbose=True, ))(other)

api_chain = APIChain.from_llm_and_api_docs(llm=chatLLM, api_docs='' ,verbose=True,)


def get_answer(question: str,user:str, chat_history: list = []):
    result = getcrc({"question": question, "chat_history": chat_history},user)
    return result

def callingApiChain(question: str, chat_history: list):
    result = getcrc({"question": question, "chat_history": chat_history})
    return result["answer"]


db = SQLDatabase.from_uri(DB_CONNECTION_STR)
toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))

agent_executor = create_sql_agent(
    llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS
)

def get_answer_from_agent(question: str, chat_history: list = []):
    result = agent_executor(question)
    return result

tools = [
    Tool(
        name="sql_agent",
        description='use to employee data from database such as active employees, performance rating, location, department, etc.',
        func=get_answer_from_agent,
        
    ),
      Tool(
        name="doc_reader",
        description='use to read resume and extract information such as name, email, phone, skills, etc., also used to read tickets and read other documents and answer ques',
        func=get_answer,
    )
]

agents = initialize_agent(tools=tools, verbose=True,  llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", verbose=True), )


def run_multiple_agents(question: str, chat_history: list = []):
    result = agents.run(question)
    return result
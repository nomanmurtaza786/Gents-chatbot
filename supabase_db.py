import os
from key_store import (SUPABASE_KEY,SUPABASE_URL,OPENAI_API_KEY,USER_ID)
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client

load_dotenv()

supabaseClient: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OpenAIEmbeddings(openai_api_key= OPENAI_API_KEY)
vector_store = SupabaseVectorStore(embedding=embeddings, client=supabaseClient, table_name="documents", query_name="match_documents")

def get_vector_store_retriever(user) :
    return vector_store.as_retriever( search_kwargs={'filter': {"user_id":user}})


def saveToSupabase(content: str, metadata: dict,):
    response = supabaseClient.table("documents").insert({"content": content, "metadata": metadata, "embedding": embeddings.embed_documents(content),"user_id":"dde6fc19-73fe-4ea6-a98b-a11485c710df"}).execute()
    
def saveDocVectorSupabase(docs: list):
     supabaseVec=vector_store.from_documents(docs, embeddings, client=supabaseClient, table_name="documents",user_id='dde6fc19-73fe-4ea6-a98b-a11485c710df') 

     
def getSimilarDocuments(text: str):
      return vector_store.similarity_search(text, 10,filter= {"user_id":"dde6fc19-73fe-4ea6-a98b-a11485c710df"})
     
     
# def getSimilarDocuments(text: str, embeddings: Embeddings):
#     return SupabaseVectorStore.similarity_search(query=text,)
    
    
# def getSimilarDocuments(text: str, embeddings: Embeddings):
#     return SupabaseVectorStore.similarity_search(query=text,)
    
    
 



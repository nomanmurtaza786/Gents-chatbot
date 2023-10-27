from key_store import (OPENAI_API_KEY,USER_ID)
import os
from dotenv import load_dotenv
from langchain.document_loaders import PDFPlumberLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from supabase_db import saveDocVectorSupabase, saveToSupabase
from PyPDF2 import PdfFileReader, PdfFileWriter,PdfReader
from langchain.schema.document import Document

embeddingOpenAi= OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

generated =False

def generateEmbeddings(pdf_bytes,user):
  global generated
  if generated == False:
          # Initialize a list to store the pages
    pages = []
    
    # Initialize PyPDF2 PdfFileReader
    pdf_reader = PdfReader(pdf_bytes)
    
    # Iterate through the pages of the PDF
    for index in range(0,len(pdf_reader.pages)):
        page = pdf_reader.pages[index]
        page_text = page.extract_text()
        
        # Create a metadata dictionary with "user_id"
        metadata = {"user_id": user,"page":index}
        
        # Append the content and metadata to the pages list
        pages.append(Document(page_content=page_text,metadata=metadata))

    # Save the pages to Supabase using your saveDocVectorSupabase function
    saveDocVectorSupabase(pages)
    print("Added into supabase successfully")
    generated = True
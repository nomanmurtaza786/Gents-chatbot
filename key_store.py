
import os
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY:str = os.getenv("OPENAI_API_KEY")
SUPABASE_URL:str  =   os.getenv("SUPABASE_URL") 
SUPABASE_KEY:str =  os.getenv("SUPABASE_KEY") 
DB_CONNECTION_STR:str = os.getenv("DB_CONNECTION_STR")
USER_ID=None
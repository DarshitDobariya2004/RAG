"""All settings for the app live here, in one place."""
import os
from dotenv import load_dotenv

load_dotenv()

##env variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

## define the path - Data and vector store
DATA_FILE_PATH = os.path.join("data","hr_policy.txt")

VECTOR_STORE_PATH = os.path.join("data","faiss_index")

#MODELS
LLM_MODEL_NAME = "openai/gpt-oss-120b"

# LLM and EMBEDDING MODEL
EMBEDDING_MODEL_NAME = "jina-embeddings-v2-base-en"

## CHUNKS / TEXT SPLITTING CONFIG

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

#RETRIEVEAL RESULTS
TOP_K_RESULTS = 3

##SYSTEM INSTRUCTION

SYSTEM_PROMPT =  ("""
You are a friendly HR assistant working for Acme Crop.Always use the search_hr_policy tool 
to look up the facts before answering.If the answer is not in the search 
results,say you do not know instead of guessing.
""")

def check_api_keys() -> None:
    """Stop early with a clear message if a required API key is missing."""
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY. Please add it to your .env file.")
    if not JINA_API_KEY:
        raise ValueError("Missing JINA_API_KEY. Please add it to your .env file.")
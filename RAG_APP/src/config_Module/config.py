# config/settings.py

from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Directories
DEFAULT_DATA_DIR = Path(
    r"C:\Projects\RAG_APP\data"
)

# Default PDF (optional - for testing)
DEFAULT_PDF_PATH = DEFAULT_DATA_DIR / "sample.pdf"

# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model configurations
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200



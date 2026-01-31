import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


class Config:
    # ===== LLM CONFIG (GROQ) =====
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant")

    # ===== RAG CONFIG =====
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    DEFAULT_URLS = ["https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2021-07-11-diffusion-models/"]

    def get_llm(self):
        """
        Returns a Groq LLM instance.
        """
        return ChatGroq(
            model=self.LLM_MODEL_NAME,
            api_key=self.GROQ_API_KEY,
            temperature=0
        )

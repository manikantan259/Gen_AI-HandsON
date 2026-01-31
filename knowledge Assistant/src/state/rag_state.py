from typing import List, Optional
from pydantic import BaseModel
from langchain_core.documents import Document


class RAGState(BaseModel):
    """
    State management for Retrieval-Augmented Generation (RAG) processes.
    This class holds the documents retrieved during the RAG process.
    """
    question: str
    answer: str = ""
    retrieved_documents: List[Document] = []
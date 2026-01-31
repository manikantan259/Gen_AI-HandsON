from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings




class VectorStore:   

    def __init__(self, persist_directory: str = "vectorstore_data"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        self.vectorstore.add_documents(documents)
        self.vectorstore.persist()

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform a similarity search in the vector store."""
        return self.vectorstore.similarity_search(query, k=k)

    def get_retriever(self):
        """Get a retriever from the vector store."""
        return self.vectorstore.as_retriever(search_kwargs={"k": 4})
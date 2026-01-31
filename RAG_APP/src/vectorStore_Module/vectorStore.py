from langchain_pinecone import PineconeVectorStore
from typing import List
from langchain_core.documents import Document



class VectorDatabase:
    """
    Handles Pinecone vector database operations.
    """

    def __init__(self, index_name: str, embeddding_model):
        self.index_name = index_name
        self.embedding_model = embeddding_model
        self.vector_store = None
    
    def load_existing_index(self):
        """
        Load an existing Pinecone index without adding documents.
        """
        try:
            self.vector_store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embedding_model
            )
            return self.vector_store
        except Exception as e:
            raise Exception(f"Failed to load existing index: {str(e)}")

    def add_documents(self, documents: List[Document]):
        """
        Adds documents to the Pinecone vector store.

        Args:
            documents (List[Document]): List of documents to add.
        """
        self.vector_store = PineconeVectorStore.from_documents(
            documents,
            self.embedding_model,
            index_name=self.index_name
        )
        return self.vector_store
            

from typing import List
from langchain_core.documents import Document


class Retrieval :



    def __init__ (self , vector_store , top_k :int =3) :

        self.vector_store = vector_store
        self.top_k = top_k

    def search (self , query : str) -> List[Document] :

        """
        Searches the vector store for relevant documents based on the query.

        Args:
            query (str): The search query. """
        


        results = self.vector_store.similarity_search(
            query,
            k=self.top_k
        )
        return results
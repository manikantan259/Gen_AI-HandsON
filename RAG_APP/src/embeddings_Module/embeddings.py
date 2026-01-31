from sentence_transformers import SentenceTransformer
from typing import List




class Embeddings:
    """ Handles text embeddings using sentence-transformers. """

    def __init__ (self , model_name :str = "sentence-transformers/all-MiniLM-L6-v2") :
        """ Initialize the embeddings module
        
        Args:
            model_name (str): Name of the embedding model to use.
        """
        self.model = SentenceTransformer(model_name)


    def embed_text(self , text : str ):

        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return [self.embed_text(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_text(text)
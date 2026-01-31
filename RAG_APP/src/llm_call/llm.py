from langchain_groq import ChatGroq
from typing import List
from langchain_core.documents import Document


class LLMMOdel:


    def __init__ (self , api_key : str , model_name : str = "llama-3.1-8b-instant") :

        self.llm = ChatGroq (
            api_key=api_key,
            model_name=model_name,
            temperature=0.7
        )
        


    def generate_answer(self, query: str, context_docs: List[Document]) -> str:

        """
        Generates a response from the LLM based on the given prompt.

        Args:
            prompt (str): The input prompt for the LLM.
            """
        
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Create prompt
        prompt = f"""Based on the following context, answer the question.
        
                Context:
                {context}

                Question: {query}

                Answer:"""
        
        # Generate response
        response = self.llm.invoke(prompt)
        return response.content


        


from src.state.rag_state import RAGState


class RAGNodes :

    def __init__(self , retriever , llm):


        self.retriever = retriever
        self.llm = llm


    def retriever_docs(self , state : RAGState) -> dict :

        docs = self.retriever.invoke(state.question)
        return {
            "retrieved_documents": docs
        }


    def generate_answer(self , state : RAGState) -> dict :

        context = "\n".join([doc.page_content for doc in state.retrieved_documents])
        prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {state.question}\n\nAnswer:"
        response = self.llm.invoke(prompt)
       
        return {
            "answer": response.content
        }
    
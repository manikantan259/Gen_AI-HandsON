from langgraph.graph import StateGraph , END
from src.state.rag_state import RAGState
from src.nodes.nodes import RAGNodes


class GraphBuilder:

    def __init__(self , retriever , llm):

        self.retriever = retriever
        self.llm = llm
        self.nodes = RAGNodes(retriever=retriever, llm=llm)
        self.graph = None

        builder  = StateGraph(RAGState)

        builder.add_node("retriever", self.nodes.retriever_docs)
        builder.add_node("responser", self.nodes.generate_answer)

        builder.set_entry_point("retriever")

        builder.add_edge("retriever", "responser")
        builder.add_edge("responser", END)

        self.graph = builder.compile()
    

    def run(self , question : str) -> RAGState :

        initial_state = RAGState(question=question)
        final_state = self.graph.invoke(initial_state)
        return final_state

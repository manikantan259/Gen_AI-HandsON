from typing import List , Optional
from asyncio import tools
from src.state.rag_state import RAGState



from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_react_agent



from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

class RAGNodes :

    def __init__(self , retriever , llm):

        self.retriever = retriever
        self.llm = llm
        self._agent = None


    def retriever_docs(self , state : RAGState) -> RAGState :

        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_documents=docs
        )
    


    def _build_agent(self) -> None :
      
        tools = self._build_tools(state=None)
        system_prompt = (
            "You are a helpful AI agent that uses tools to answer questions. "
            "Prefer retriever for the user provided docs use wikipedia for general knowledge. "
            "Provide accurate answers."
        )
        self._agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=system_prompt,
            verbose=True
        )



    def _build_tools(self , state : RAGState) -> List[Tool] :

        retriever_tool = Tool(
            name="Retriever",
            func=lambda query: "\n".join([doc.page_content for doc in state.retrieved_documents]),
            description="Useful for answering questions based on the provided documents."
        )

        wikipedia = WikipediaAPIWrapper(top_k_results=3 , lang="en")
        wikipedia_tool = WikipediaQueryRun(wikipedia_api_wrapper=wikipedia)

        return [retriever_tool, wikipedia_tool]
    


    def generate_answer(self , state : RAGState) -> RAGState :

        if self._agent is None :
            self._build_agent()

        tools = self._build_tools(state=state)
        self._agent.tools = tools

        user_message = HumanMessage(content=state.question)
        response = self._agent.invoke([user_message])

        return RAGState(
            question=state.question,
            retrieved_documents=state.retrieved_documents,
            answer=response.content or "Could not find an answer."
        )




        























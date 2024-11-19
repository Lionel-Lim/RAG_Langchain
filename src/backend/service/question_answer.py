import yaml
from typing import List
from typing_extensions import TypedDict
from typing import Annotated

from flask import Config
from langchain_google_vertexai import ChatVertexAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from backend.service.vector_store import VectorStore


class State(TypedDict):
    messages: Annotated[List[SystemMessage | HumanMessage | AIMessage], add_messages]


class Chatbot:
    def __init__(self, config: Config, vector_store: VectorStore):
        self.config = config
        self.vector_store = vector_store
        self.llm = ChatVertexAI(model=config["llm_model"])
        self.prompts = self.load_prompts()
        self.graph = self.build_graph()

    @staticmethod
    def load_prompts():
        with open("common/prompts.yaml", "r") as file:
            return yaml.safe_load(file)

    def build_graph(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", self.chatbot)
        graph_builder.set_entry_point("chatbot")
        graph_builder.set_finish_point("chatbot")
        return graph_builder.compile()

    async def chatbot(self, state: State):
        response = await self.llm.ainvoke(state["messages"])
        state["messages"].append(response)
        return state

    async def ask_question(self, question: str):
        try:
            related_chunks = await self.get_related_chunks(question)
            if not related_chunks:
                return {"error": "No related chunks found"}

            context_str = "\n".join(related_chunks)
            prompt_template = self.prompts["context_prompt"]["template"]
            prompt = prompt_template.format(context_str=context_str, query_str=question)

            initial_state = {
                "messages": [
                    SystemMessage(content=prompt),
                    HumanMessage(content=question),
                ]
            }

            final_state = await self.graph.ainvoke(initial_state)
            response_message = final_state["messages"][-1]
            return {"answer": response_message.content}
        except Exception as e:
            self.log_error(f"Error answering question: {str(e)}")
            return {"error": "Failed to answer question"}

    async def get_related_chunks(self, query: str) -> List[str]:
        try:
            related_documents = self.vector_store.search_documents(query)
            return [doc.page_content for doc in related_documents]
        except Exception as e:
            self.log_error(f"Error retrieving related chunks: {str(e)}")
            return []

    def log_error(self, message: str):
        # Implement proper logging here, e.g., using Python's logging module
        print(f"ERROR: {message}")


# Usage example:
# async def main():
#     config = Config(...)  # Initialize with appropriate configuration
#     vector_store = VectorStore(...)  # Initialize your vector store
#     chatbot = Chatbot(config, vector_store)
#     response = await chatbot.ask_question("What is the weather in San Francisco?")
#     print(response)

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())

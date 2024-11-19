from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated
from flask import Config
from langchain.schema import Document
from service.vector_store import VectorStore
import yaml


class State(TypedDict):
    messages: Annotated[list, add_messages]


class Chatbot:
    def __init__(self, config: Config, vector_store: VectorStore):
        GOOGLE_CLOUD_EXTERNAL_IP = config["google_cloud_external_ip"]
        LLM_MODEL = config["llm_model"]

        self.llm = ChatVertexAI(model=LLM_MODEL)
        self.vector_store = vector_store
        self.graph_builder = StateGraph(State)
        self.graph_builder.add_node("chatbot", self.chatbot)
        self.graph_builder.set_entry_point("chatbot")
        self.graph_builder.set_finish_point("chatbot")
        self.graph = self.graph_builder.compile()

        # Load the prompt template from prompts.yaml
        with open("common/prompts.yaml", "r") as file:
            self.prompts = yaml.safe_load(file)

    def chatbot(self, state: State):
        response = self.llm.invoke(state["messages"])
        return {"messages": [response]}

    def ask_question(self, question):
        try:
            # Retrieve related chunks from the vector store
            related_chunks = self.get_related_chunks(question)
            if not related_chunks:
                return {"error": "No related chunks found"}

            # Prepare the context string from the related chunks
            context_str = "\n".join(related_chunks)

            # Prepare the prompt using the template
            prompt_template = self.prompts["context_prompt"]["template"]
            prompt = prompt_template.format(context_str=context_str, query_str=question)

            # Prepare the messages with the prompt
            messages = [{"role": "system", "content": prompt}]

            # Ask the LLM with the prepared messages
            initial_state = {"messages": messages}
            final_state = self.graph.invoke(
                initial_state
            )  # Use the correct method to execute the state graph
            response_message = final_state["messages"][-1]
            return {"answer": response_message}
        except Exception as e:
            print(f"Error answering question: {str(e)}")
            return {"error": "Failed to answer question"}

    def get_related_chunks(self, query: str) -> list[str]:
        try:
            # Use the vector store to find related chunks
            related_documents = self.vector_store.search_documents(query)
            return [doc.page_content for doc in related_documents]
        except Exception as e:
            print(f"Error retrieving related chunks: {str(e)}")
            return []

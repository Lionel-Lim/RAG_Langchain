# export PYTHONPATH="."

import os
import sys

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from src.common.utils import load_yaml

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.prompts import PromptTemplate

# Set your GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "env/ai-sandbox-company-73-2659f4150720.json"
)
# Set up the USER_AGENT environment variable
os.environ["USER_AGENT"] = "myagent"

# Load prompt templates
prompts_file_path = r"src/backend/src/prompts.yaml"
prompts = load_yaml(prompts_file_path)

# Load the config file
config_file_path = r"src/backend/config.yaml"
config = load_yaml(config_file_path)

project = config["project_id"]
location = config["location"]
llm_model = config["llm_model"]
embeddings_model = config["embeddings_model"]

# Initialize the Vertex AI model and embeddings
llm = ChatVertexAI(model=llm_model)
embeddings = VertexAIEmbeddings(model_name=embeddings_model)

# Load, chunk and index the document
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=768, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

collection_name = "demo_collection"

client = QdrantClient(":memory:")
# client = QdrantClient(path="/tmp/langchain_qdrant")

# Define a collection to store your vectors.
# Ensure the size parameter matches the dimensionality of your embeddings (e.g., 768 for Vertex AI embeddings):
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)
vector_store.add_documents(documents=splits)

retriever = vector_store.as_retriever()

# Create a PromptTemplate
context_prompt = PromptTemplate(
    template=prompts["context_prompt"]["template"],
    input_variables=prompts["context_prompt"]["variables"],
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context_str": retriever | format_docs, "query_str": RunnablePassthrough()}
    | context_prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("What is Task Decomposition?"))

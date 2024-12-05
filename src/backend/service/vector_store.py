from qdrant_client import QdrantClient
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from langchain.schema import Document  # Make sure to import Document
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

import logging


class VectorStore:
    def __init__(self, config):
        DB_SERVICE_NAME = config["db_service_name"]
        EMBEDDINGS_MODEL = config["embeddings_model"]
        VECTORSTORE_COLLECTION_NAME = config["vectorstore_collection_name"]
        ADVANCED_SEARCH_COLLECTION_NAME = config["advanced_search_collection_name"]

        embeddings = VertexAIEmbeddings(model_name=EMBEDDINGS_MODEL)
        client = QdrantClient(url=f"{DB_SERVICE_NAME}:6334", prefer_grpc=True)

        # Create the collection if it does not exist
        if not client.collection_exists(VECTORSTORE_COLLECTION_NAME):
            client.create_collection(
                collection_name=VECTORSTORE_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=768, distance=Distance.COSINE, on_disk=True
                ),
            )
            logging.info(f"Created new collection: {VECTORSTORE_COLLECTION_NAME}")
        else:
            logging.info(f"Using existing collection: {VECTORSTORE_COLLECTION_NAME}")
        # Create the advanced search collection if it does not exist
        if not client.collection_exists(ADVANCED_SEARCH_COLLECTION_NAME):
            client.create_collection(
                collection_name=ADVANCED_SEARCH_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=768, distance=Distance.COSINE, on_disk=True
                ),
            )
            logging.info(f"Created new collection: {ADVANCED_SEARCH_COLLECTION_NAME}")
        else:
            logging.info(
                f"Using existing collection: {ADVANCED_SEARCH_COLLECTION_NAME}"
            )

        # Initialize the vector store
        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name=VECTORSTORE_COLLECTION_NAME,
            embedding=embeddings,
        )
        self.advanced_search_vector_store = QdrantVectorStore(
            client=client,
            collection_name=ADVANCED_SEARCH_COLLECTION_NAME,
            embedding=embeddings,
        )

        self.client = client

    def add_documents(
        self, documents: list[Document], is_advance_search: bool = False
    ) -> list[str]:
        try:
            # Call the add_documents method and capture the returned IDs
            if is_advance_search:
                ids = self.advanced_search_vector_store.add_documents(documents)
            else:
                ids = self.vector_store.add_documents(documents)
            logging.info(f"Added {len(ids)} documents to the vector store.")
            return ids  # Return the IDs if needed
        except ValueError as ve:
            # Handle the ValueError if the number of IDs doesn't match the number of documents
            logging.error(f"ValueError adding documents: {ve}")
            raise
        except Exception as e:
            # Handle other potential exceptions
            logging.error(f"Error adding documents: {e}")
            raise

    def search_documents(
        self,
        query: str,
        is_advance_search: bool = False,
        document_guids: list[str] = [],
    ) -> list[Document]:
        try:
            # Create a filter if document_guids are provided
            filter_condition = None
            if document_guids:
                filter_condition = Filter(
                    must=[
                        FieldCondition(
                            key="metadata.file_guid", match=MatchValue(value=guid)
                        )
                        for guid in document_guids
                    ]
                )

            # Search for related documents in the vector store
            if is_advance_search:
                results = self.advanced_search_vector_store.similarity_search(
                    query, filter=filter_condition
                )
            else:
                results = self.vector_store.similarity_search(query)

            return results
        except Exception as e:
            logging.error(f"Error searching documents: {e}")
            raise

    # def get_all_document_names(self):
    #     document_names = set()
    #     try:
    #         # scroll_result = self.client.scroll(
    #         #     collection_name=self.vector_store.collection_name
    #         # )
    #         # for point in scroll_result:
    #         #     # Ensure point is a dictionary and has 'payload'
    #         #     if isinstance(point, dict) and "payload" in point:
    #         #         # Extract the document name from the payload
    #         #         document_name = point["payload"].get("name")
    #         #         if document_name:  # Check if name exists
    #         #             document_names.append(document_name)

    #         # Scroll through all points
    #         scroll_filter = Filter(
    #             must=[
    #                 FieldCondition(
    #                     key="document_name",
    #                     match=MatchValue(value="*")  # Match all document names
    #                 )
    #             ]
    #         )
    #     except Exception as e:
    #         logging.error(f"Error getting document names: {e}")
    #         raise
    #     return document_names

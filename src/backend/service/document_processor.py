from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from backend.service.vector_store import VectorStore
from langchain.schema import Document
import logging


class DocumentProcessor:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def process_document(self, file):
        try:
            temp_path = f"/tmp/{file.filename}"
            file.save(temp_path)
            loader = PyPDFLoader(temp_path)
            documents = loader.load()  # This should return a list of Documents

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)

            # Ensure that splits is a list of Document instances
            if not all(isinstance(doc, Document) for doc in splits):
                raise ValueError("All items in splits must be Document instances.")

            # Call add_documents and handle the returned IDs
            ids = self.vector_store.add_documents(splits)
            logging.info(f"Document {file.filename} processed")

            return True
        except ValueError as ve:
            logging.error(f"ValueError processing document {file.filename}: {ve}")
            return False
        except Exception as e:
            logging.error(f"Error processing document {file.filename}: {e}")
            return False

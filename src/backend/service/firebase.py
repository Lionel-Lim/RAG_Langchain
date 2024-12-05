from typing import List
from model.pdf_metadata import DocumentMetadata
from google.cloud import firestore
from google.oauth2 import service_account


class FirestoreManager:
    def __init__(self):
        self.credential = service_account.Credentials.from_service_account_file(
            r"credential/ai-sandbox-company-73-2659f4150720.json"
        )
        self.db = firestore.Client(credentials=self.credential)

    def get_all_document_metadata(self):
        """
        Retrieves all document metadata from Firestore.
        """
        collection_ref = self.db.collection("advanced_collection")
        docs = collection_ref.stream()

        all_docs = []
        for doc in docs:
            data = doc.to_dict()
            all_docs.append(data)

        return all_docs

    def save_document_metadata_advanced(self, doc_meta: DocumentMetadata):
        """
        Saves detailed parsed document metadata and its pages/elements into Firestore.
        """

        # Prepare the main document data
        doc_data = {
            "file_name": doc_meta.file_name,
            "file_guid": doc_meta.file_guid,
            "file_type": doc_meta.file_type,
            "file_size": doc_meta.file_size,
        }

        # Reference to the main document using file_guid
        doc_ref = self.db.collection("advanced_collection").document(doc_meta.file_guid)

        # Set the main document data
        doc_ref.set(doc_data)

        # Save each page as a separate document in a 'pages' subcollection
        for page in doc_meta.pages:
            page_data = {
                "page_number": page.page_number,
                "metadata": page.metadata,
                "elements": [element.model_dump() for element in page.elements],
            }
            # Reference to the page document in the 'pages' subcollection
            page_ref = doc_ref.collection("pages").document(str(page.page_number))
            # Set the page document data
            page_ref.set(page_data)

    def save_document_metadata_simple(self, doc_guid: str, vector_ids: List[str]):
        """
        Saves simpler document references (like Qdrant vector IDs) into Firestore.
        """
        main_doc_data = {
            "doc_guid": doc_guid,
            "vectors": {
                str(idx): {"node_id": v_id} for idx, v_id in enumerate(vector_ids)
            },
        }
        doc_ref = self.db.collection("simple_collection").document(doc_guid)

        doc_ref.set(main_doc_data)

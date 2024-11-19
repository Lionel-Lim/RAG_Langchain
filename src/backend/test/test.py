from src.backend.service.vector_store import VectorStore
from src.common.utils import load_yaml
import os

# Load configuration settings
config_file_path = r"src/backend/config.yaml"
config = load_yaml(config_file_path)

# Set environment variable for Google Application Credentials
SERVICE_ACCOUNT_ADDRESS = config["service_account_address"]
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_ADDRESS

vector_store = VectorStore(config)

# List all documents
all_documents = vector_store.get_all_document_names()

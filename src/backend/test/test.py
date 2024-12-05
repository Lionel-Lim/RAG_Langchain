from typing import List
from google.cloud import firestore
from google.oauth2 import service_account
import json

credential = service_account.Credentials.from_service_account_file(
    r"./credential/ai-sandbox-company-73-2659f4150720.json"
)

db = firestore.Client(credentials=credential)

collection_ref = db.collection("advanced_collection")

docs = collection_ref.stream()

# Extract 'file_name' from each document
file_names = []
for doc in docs:
    data = doc.to_dict()
    file_names.append(data)

# Convert the list of file names to JSON format
file_names_json = json.dumps(file_names, indent=2)
print(file_names[0])

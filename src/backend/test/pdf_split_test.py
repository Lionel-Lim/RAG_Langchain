"""
cd /Users/dylim/Documents/projects/RAG_Langchain/src/backend/
export PYTHONPATH="."
python src/backend/test/pdf_split_test.py
"""

from service.document_processor import DocumentProcessor
import os
import re

document_processor = DocumentProcessor()


pdf_path = r"/Users/dylim/Documents/projects/RAG_Langchain/src/backend/test/pdf_finaltest/CP 83-1-2004 (2015) CoP for construction CAD - Organisation n naming of CAD layers.pdf"

result = document_processor.split_pdf_if_needed(pdf_path)

print(result)


# test_pdf_parse = r"/Users/dylim/Documents/projects/RAG_Langchain/src/backend/test/pdf_sample/test.pdf"


def get_pdf_paths(directory):
    pdf_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pdf"):
                pdf_paths.append(os.path.join(root, file))
    return pdf_paths


def extract_serial_number(file_path):
    # Extract the serial number part from the file name
    match = re.search(r"_(\d+)\.pdf$", os.path.basename(file_path))
    return int(match.group(1)) if match else float("inf")


# Directory containing the PDFs
# pdf_directory = (
#     r"/Users/dylim/Documents/projects/RAG_Langchain/src/backend/test/pdf_test"
# )

# Get all PDF paths
# pdf_paths = get_pdf_paths(pdf_directory)

# Sort the PDF paths based on the serial number part of the file names
# pdf_paths.sort(key=extract_serial_number)


document_processor.parse_document(result)

document_processor.create_summary()

print(document_processor)

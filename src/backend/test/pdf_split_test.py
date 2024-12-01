"""
cd /Users/dylim/Documents/projects/RAG_Langchain/src/backend/
export PYTHONPATH="."
python src/backend/test/pdf_split_test.py
"""

from service.document_processor import DocumentProcessor

document_processor = DocumentProcessor()


# pdf_path = r"/Users/dylim/Documents/projects/RAG_Langchain/src/backend/test/pdf_sample/CORENET X Code of Practice.pdf"

# result = document_processor.split_pdf_if_needed(pdf_path)

# print(result)


test_pdf_parse = r"/Users/dylim/Documents/projects/RAG_Langchain/src/backend/test/pdf_sample/test.pdf"

document_processor.parse_document([test_pdf_parse])

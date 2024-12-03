from typing import List, Dict, Any, Union
from pydantic import BaseModel


class PDFMetadata(BaseModel):
    # Define the fields in split_pdf_if_needed function
    file_name: str = ""
    file_guid: str = ""
    file_type: str = "pdf"
    file_size: int = 0
    page_numbers: int = 0

    # Define the fields in parse_document function
    # batch_size: int = 0
    # split_file_paths: List[str] = []
    # analyzed_file_paths: List[str] = []
    page_elements: Dict[int, Dict[int, str]] = {}  # {page_number: {element_id: 'type'}}
    page_metadata: Dict[int, Dict] = {}  # {page_number: {...}}
    page_summaries: Dict[int, str] = {}  # {page_number: 'summary'}
    image_paths: Dict[int, Dict[str, str]] = {}  # {tag_id: ['type', 'path']}
    image_summaries: Dict[int, str] = {}  # {tag_id: 'summary'}
    table_string: Dict[int, str] = {}  # {tag_id: 'markdown'}
    table_summaries: Dict[int, str] = {}  # {tag_id: 'summary'}
    texts: Dict[int, Dict[int, str]] = {}  # {tag_id: 'text'}
    text_summaries: Dict[str, str] = {}  # {tag_id: 'text'}
    pdf_gcs_uri: str = ""  # GCS URI for the PDF file


class Element(BaseModel):
    element_id: int
    type: str


class ImageElement(Element):
    path: str = ""
    summary: str = ""


class TableElement(Element):
    path: str = ""
    raw_text: str = ""
    summary: str = ""


class TextElement(Element):
    text: str = ""
    summary: str = ""


class Page(BaseModel):
    page_number: int
    elements: List[Union[ImageElement, TableElement, TextElement]] = []
    metadata: Dict[str, Any] = {}
    summary: str = ""


class DocumentMetadata(BaseModel):
    file_name: str = ""
    file_guid: str = ""
    file_type: str = "pdf"
    file_size: int = 0
    page_numbers: int = 0
    pdf_gcs_uri: str = ""
    pages: List[Page] = []

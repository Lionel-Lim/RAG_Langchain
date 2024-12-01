from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from service.vector_store import VectorStore
from langchain_upstage import UpstageDocumentParseLoader
from common.utils import load_yaml, load_json
from typing import List
from pypdf import PdfReader, PdfWriter
from bs4 import BeautifulSoup
from model.pdf_metadata import (
    PDFMetadata,
    DocumentMetadata,
    Page,
    TextElement,
    ImageElement,
    TableElement,
)
from PIL import Image

import os
import hashlib
import uuid
import pymupdf
import requests
import logging


class DocumentProcessor:
    def __init__(self):
        self.parsed_metadata = PDFMetadata()
        self.metadata = DocumentMetadata()

    def process_document(self, vector_store: VectorStore, file):
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
            ids = vector_store.add_documents(splits)
            logging.info(f"Document {file.filename} processed")

            return True
        except ValueError as ve:
            logging.error(f"ValueError processing document {file.filename}: {ve}")
            return False
        except Exception as e:
            logging.error(f"Error processing document {file.filename}: {e}")
            return False

    def parse_document(self, file_path_list: List[str]):
        # Load api key
        config_file_path = r"./config.yaml"
        config = load_yaml(config_file_path)

        # Open and read the JSON file
        API_KEY = load_json(config["api_key_address"])["upstage_key"]
        URL = r"https://api.upstage.ai/v1/document-ai/document-parse"
        headers = {"Authorization": f"Bearer {API_KEY}"}

        # Initialize variables for multiple files
        last_page_num = 0
        last_tag_id = 0

        for file_path in file_path_list:
            doc = {"document": open(file_path, "rb")}
            response = requests.post(URL, headers=headers, files=doc)
            result = response.json()

            current_page_num = 0

            for elem in result["elements"]:
                if current_page_num != elem["page"] + last_page_num:
                    page = Page(page_number=current_page_num)
                    current_page_num = elem["page"] + last_page_num

                element_id = elem["id"] + last_tag_id
                element_category = elem["category"]

                if element_category in [
                    "heading1",
                    "caption",
                    "paragraph",
                    "equation",
                    "list",
                    "index",
                ]:
                    parsed_tag = BeautifulSoup(elem["content"]["html"], "html.parser")
                    page.elements.append(
                        TextElement(
                            element_id=element_id,
                            text=parsed_tag.get_text(),
                            type=element_category,
                        )
                    )
                # TODO: Process table and image elements
                elif element_category in ["table", "figure", "chart"]:
                    img = self.pdfpage_to_image(file_path, current_page_num)
                    img_path = os.path.join(
                        os.path.dirname(file_path),
                        "img",
                        str(self.parsed_metadata.file_guid),
                        f"{current_page_num}_{element_id}_{element_category}.png",
                    )
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    self.crop_image(img, elem["coordinates"], img_path)
                    if element_category == "table":
                        page.elements.append(
                            TableElement(
                                element_id=element_id,
                                path=img_path,
                                raw_text=elem["content"]["text"],
                                type=element_category,
                            )
                        )

            last_page_num = current_page_num
            last_tag_id = element_id

    def pdfpage_to_image(self, file_path: str, page_num: int, dpi=300):
        with pymupdf.open(file_path) as doc:
            page = doc[page_num - 1].get_pixmap(dpi=dpi)
            target_page_size = [page.width, page.height]
            page_img = Image.frombytes("RGB", target_page_size, page.samples)
        return page_img

    def crop_image(self, img: Image, coordinates: list, output_path: str):
        # Get the image size
        width, height = img.size

        # Extract all x and y values
        x_values = [point["x"] for point in coordinates]
        y_values = [point["y"] for point in coordinates]

        # Find the min and max values to define the bounding box
        x1, x2 = min(x_values), max(x_values)
        y1, y2 = min(y_values), max(y_values)

        # Convert to pixel values
        x1, y1, x2, y2 = [
            int(coord * dim)
            for coord, dim in zip([x1, y1, x2, y2], [width, height, width, height])
        ]

        # Crop the image
        crop_box = (x1, y1, x2, y2)
        cropped_img = img.crop(crop_box)

        # Save the cropped image
        cropped_img.save(output_path)

    # def descrbe_image(self, image, context):

    # Function to split PDF if it exceeds API limitations
    def split_pdf_if_needed(self, file_path: str) -> List[str]:
        self.parsed_metadata.file_guid = self.get_doc_guid(file_path)
        self.metadata.file_guid = str(self.parsed_metadata.file_guid)
        file_guid = self.metadata.file_guid

        self.parsed_metadata.file_name = os.path.basename(file_path)
        self.metadata.file_name = self.parsed_metadata.file_name

        file_dir = os.path.dirname(file_path)

        reader = PdfReader(file_path)

        total_pages = len(reader.pages)
        self.parsed_metadata.page_numbers = total_pages
        self.metadata.page_numbers = total_pages

        # Check file size (in MB)
        self.parsed_metadata.file_size = os.path.getsize(file_path)
        self.metadata.file_size = os.path.getsize(file_path)
        file_size_mb = self.parsed_metadata.file_size / (1024 * 1024)

        split_files = []
        if total_pages > 100 or file_size_mb > 50:
            # Calculate number of splits needed
            num_splits = max(
                (total_pages // 100) + (total_pages % 100 > 0),
                int(file_size_mb // 50) + 1,
            )
            pages_per_split = total_pages // num_splits + (total_pages % num_splits > 0)
            for i in range(num_splits):
                writer = PdfWriter()
                start_page = i * pages_per_split
                end_page = min(start_page + pages_per_split, total_pages)
                for page in reader.pages[start_page:end_page]:
                    writer.add_page(page)
                file_name = f"{file_guid}_{i}.pdf"
                split_file_path = os.path.join(file_dir, file_name)
                with open(split_file_path, "wb") as f:
                    writer.write(f)
                split_files.append(split_file_path)
        else:
            split_files.append(file_path)

        # Test Only
        if True:
            writer = PdfWriter()
            for page in [reader.pages[0], *reader.pages[4:30], reader.pages[-1]]:
                writer.add_page(page)
            split_file_path = os.path.join(file_dir, "test.pdf")
            with open(split_file_path, "wb") as f:
                writer.write(f)

        return split_files

    def get_doc_guid(self, file_path: str) -> str:
        # Initialize a hash object
        hash_md5 = hashlib.md5()

        # Read the PDF file
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                # Update the hash object with the text content
                hash_md5.update(text.encode("utf-8"))

        # Generate a UUID using the MD5 hash
        pdf_uuid = uuid.UUID(hash_md5.hexdigest())
        return pdf_uuid

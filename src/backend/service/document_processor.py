from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSerializable
from langchain.schema import Document
from service.vector_store import VectorStore
from common.utils import load_yaml, load_json
from typing import List, Union
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
from model.pdf_metadata import (
    ImageElement,
    TableElement,
    TextElement,
    Page,
    DocumentMetadata,
)
from langchain_google_vertexai import ChatVertexAI
from service.firebase import FirestoreManager


import os
import hashlib
import uuid
import pymupdf
import requests
import logging
import json
import yaml
import base64

ChatPromptTemplate.from_template
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level to DEBUG
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output logs to the console
        logging.FileHandler("debug.log"),  # Optionally, output logs to a file
    ],
)


class DocumentProcessor:
    def __init__(self):
        self.metadata = DocumentMetadata()

    def process_document(self, vector_store: VectorStore, file):
        try:
            temp_path = f"/tmp/{file.filename}"
            file.save(temp_path)
            loader = PyPDFLoader(temp_path)
            documents = loader.load()  # This should return a list of Documents

            doc_guid = str(self.get_doc_guid(temp_path))

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)

            # Ensure that splits is a list of Document instances
            if not all(isinstance(doc, Document) for doc in splits):
                raise ValueError("All items in splits must be Document instances.")

            # Call add_documents and handle the returned IDs
            ids = vector_store.add_documents(splits)

            # Save the metadata to Firestore
            firestore_manager = FirestoreManager()
            firestore_manager.save_document_metadata_simple(doc_guid, ids)

            logging.info(f"Document {file.filename} processed")

            return True
        except ValueError as ve:
            logging.error(f"ValueError processing document {file.filename}: {ve}")
            return False
        except Exception as e:
            logging.error(f"Error processing document {file.filename}: {e}")
            return False

    def advanced_process_document(self, vector_store: VectorStore) -> bool:
        all_nodes: List[Document] = []
        if not self.metadata.file_guid:
            return False
        try:
            # Save page level summary
            for page in self.metadata.pages:
                if page.summary:
                    node = Document(
                        page_content=page.summary,
                        metadata={
                            "page_number": page.page_number,
                            "file_guid": self.metadata.file_guid,
                        },
                    )
                    all_nodes.append(node)
            # Save element level summary
            for page in self.metadata.pages:
                for element in page.elements:
                    if element.summary:
                        node = Document(
                            page_content=element.summary,
                            metadata={
                                "page_number": page.page_number,
                                "element_id": element.element_id,
                                "file_guid": self.metadata.file_guid,
                            },
                        )
                        all_nodes.append(node)
            vector_store.add_documents(all_nodes, is_advance_search=True)

            # Save metadata to Firestore
            firestore_manager = FirestoreManager()
            firestore_manager.save_document_metadata_advanced(self.metadata)

            logging.info(f"Document {self.metadata.file_name} processed")
            return True
        except ValueError as ve:
            logging.error(
                f"ValueError processing document {self.metadata.file_name}: {ve}"
            )
            return False
        except Exception as e:
            logging.error(f"Error processing document {self.metadata.file_name}: {e}")
            return False

    def parse_document(self, file_path_list: List[str]):
        # Load API key
        config_file_path = r"./config.yaml"
        config = load_yaml(config_file_path)
        API_KEY = load_json(config["api_key_address"])["upstage_key"]
        URL = r"https://api.upstage.ai/v1/document-ai/document-parse"
        headers = {"Authorization": f"Bearer {API_KEY}"}

        # Initialize document GUID if not already set
        if not self.metadata.file_guid:
            self.metadata.file_guid = str(uuid.uuid4())
        document_guid = self.metadata.file_guid

        # Initialize the list of pages if not already initialized
        if not self.metadata.pages:
            self.metadata.pages = []

        # Variables to keep track of cumulative page numbers and element IDs
        cumulative_page_num = 0
        # Initialize element counter for generating sequential IDs
        element_counter = 0

        for file_path in file_path_list:
            # Open the PDF document once per file
            with pymupdf.open(file_path) as pdf_doc:
                # Send the document for parsing
                with open(file_path, "rb") as f:
                    doc = {"document": f}
                    response = requests.post(URL, headers=headers, files=doc)
                    result = response.json()

                # Initialize variables for tracking current page and elements
                current_page_num = None
                page = None

                for elem in result["elements"]:
                    # Update page number by adding cumulative_page_num
                    elem_page_num = elem["page"] + cumulative_page_num

                    # Check if we have moved to a new page
                    if current_page_num != elem_page_num:
                        # Save the previous page
                        if page is not None:
                            self.metadata.pages.append(page)
                        # Start a new page
                        page = Page(page_number=elem_page_num)
                        current_page_num = elem_page_num

                    # Update element ID by adding cumulative_element_id
                    element_id = element_counter

                    element_category = elem["category"]

                    if element_category in [
                        "heading1",
                        "caption",
                        "paragraph",
                        "equation",
                        "list",
                        "index",
                    ]:
                        parsed_tag = BeautifulSoup(
                            elem["content"]["html"], "html.parser"
                        )
                        page.elements.append(
                            TextElement(
                                element_id=element_id,
                                text=parsed_tag.get_text(),
                                type=element_category,
                            )
                        )
                        element_counter += 1
                    elif element_category in ["table", "figure", "chart"]:
                        # Convert PDF page to image
                        img = self.pdfpage_to_image(pdf_doc, elem["page"])
                        img_path = os.path.join(
                            os.path.dirname(file_path),
                            "img",
                            document_guid,
                            f"{elem_page_num}_{element_id}_{element_category}.png",
                        )
                        os.makedirs(os.path.dirname(img_path), exist_ok=True)
                        # Crop and save the image
                        self.crop_image(img, elem["coordinates"], img_path)

                        if element_category == "table":
                            page.elements.append(
                                TableElement(
                                    element_id=element_id,
                                    path=img_path,
                                    raw_text=elem["content"]["html"],
                                    type=element_category,
                                )
                            )
                        else:
                            page.elements.append(
                                ImageElement(
                                    element_id=element_id,
                                    path=img_path,
                                    type=element_category,
                                )
                            )
                        element_counter += 1

                # Append the last page after processing all elements in the file
                if page is not None:
                    self.metadata.pages.append(page)

                # Update cumulative counters
                # Assuming that 'page' variable now holds the last page processed
                if page is not None:
                    cumulative_page_num = page.page_number

        # Update metadata fields
        self.metadata.file_name = os.path.basename(
            file_path_list[0]
        )  # Assuming all files are from the same document
        self.metadata.file_size = sum(os.path.getsize(fp) for fp in file_path_list)
        self.metadata.page_numbers = cumulative_page_num

    def pdfpage_to_image(self, file_path: str, page_num: int, dpi=300) -> Image:
        """
        Convert a PDF page to an image using pymupdf

        Args:
            file_path (str): Path to the PDF file
            page_num (int): Page number to convert to image
            dpi (int): Resolution of the image

        Returns:
            Image: Image object
        """
        with pymupdf.open(file_path) as doc:
            page = doc[page_num - 1].get_pixmap(dpi=dpi)
            target_page_size = [page.width, page.height]
            page_img = Image.frombytes("RGB", target_page_size, page.samples)
        return page_img

    def crop_image(self, img: Image, coordinates: list, output_path: str) -> None:
        """

        Crop an image using a list of coordinates

        Args:
            img (Image): Image object
            coordinates (list): List of coordinates
            output_path (str): Path to save the cropped image
        """
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
        self.metadata.file_guid = str(self.get_doc_guid(file_path))
        file_guid = self.metadata.file_guid

        self.metadata.file_name = os.path.basename(file_path)

        file_dir = os.path.dirname(file_path)

        # Open the PDF with pymupdf
        doc = pymupdf.open(file_path)

        total_pages = len(doc)
        self.metadata.page_numbers = total_pages

        # Check file size (in MB)
        self.metadata.file_size = os.path.getsize(file_path)
        file_size_mb = self.metadata.file_size / (1024 * 1024)

        split_files = []
        if total_pages > 100 or file_size_mb > 50:
            # Calculate number of splits needed
            num_splits = max(
                (total_pages // 100) + (total_pages % 100 > 0),
                int(file_size_mb // 50) + 1,
            )
            pages_per_split = total_pages // num_splits + (total_pages % num_splits > 0)
            for i in range(num_splits):
                start_page = i * pages_per_split
                end_page = min(start_page + pages_per_split, total_pages)

                # Create a new PDF
                new_doc = pymupdf.open()  # Empty document
                new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)

                file_name = f"{file_guid}_{i}.pdf"
                split_file_path = os.path.join(file_dir, file_name)
                new_doc.save(split_file_path)
                split_files.append(split_file_path)
                new_doc.close()
        else:
            split_files.append(file_path)

        # Test Only
        if True:
            writer = pymupdf.open()  # Create a new empty PDF
            for page_num in [0, *range(4, 30), len(doc) - 1]:
                writer.insert_pdf(doc, from_page=page_num, to_page=page_num)
            split_file_path = os.path.join(file_dir, "test.pdf")
            writer.save(split_file_path)
            writer.close()

        doc.close()
        return split_files

    def get_doc_guid(self, file_path: str) -> str:
        # Initialize a hash object
        hash_md5 = hashlib.md5()

        # Read the PDF file using pymupdf
        with pymupdf.open(file_path) as doc:
            for page in doc:
                text = page.get_text()
                if text:
                    # Update the hash object with the text content
                    hash_md5.update(text.encode("utf-8"))

        # Generate a UUID using the MD5 hash
        pdf_uuid = uuid.UUID(hash_md5.hexdigest())
        return pdf_uuid

    def create_summary(self):
        logging.info("Starting create_summary method.")

        # Load prompt templates
        logging.debug("Loading prompt templates from common/prompts.yaml.")
        with open("common/prompts.yaml", "r") as file:
            prompts = yaml.safe_load(file)
        prompt_template_image = prompts["image_to_text_prompt"]["template"]
        prompt_template_text = PromptTemplate.from_template(
            prompts["text_summarization_prompt"]["template"]
        )
        logging.info("Prompt templates loaded successfully.")

        # Flatten Elements across pages
        logging.debug("Flattening elements across all pages.")
        all_elements = []
        for page in self.metadata.pages:
            all_elements.extend(page.elements)
        logging.info(f"Total elements flattened: {len(all_elements)}.")

        # Ensure all Elements are sorted by element_id
        logging.debug("Sorting all elements by element_id.")
        all_elements.sort(key=lambda x: x.element_id)
        logging.info("Elements sorted successfully.")

        # Initialize AI model
        logging.debug("Initializing ChatVertexAI model.")
        chat_model = ChatVertexAI(model="gemini-1.5-pro-002", temperature=0.1)
        fast_model = ChatVertexAI(model="gemini-1.5-flash-002", temperature=0.1)
        logging.info("ChatVertexAI model initialized.")

        # Define prompt templates and chains
        logging.debug("Defining prompt templates and chains.")
        text_chain = prompt_template_text | chat_model | StrOutputParser()
        image_chain = chat_model | StrOutputParser()
        logging.info("Prompt templates and chains defined.")

        # Process image elements
        logging.info("Identifying image and table elements for processing.")
        image_elements = [
            element
            for element in all_elements
            if isinstance(element, (ImageElement, TableElement))
        ]
        logging.info(f"Found {len(image_elements)} image/table elements.")

        # Process text elements
        logging.info("Identifying text elements for processing.")
        text_elements = [
            element for element in all_elements if isinstance(element, TextElement)
        ]
        logging.info(f"Found {len(text_elements)} text elements.")

        # Batch processing
        logging.info("Starting batch processing of image elements.")
        self.batch_process_images(
            image_elements, all_elements, fast_model, prompt_template_image
        )
        logging.info("Completed batch processing of image elements.")

        logging.info("Starting batch processing of text elements.")
        self.batch_process_texts(text_elements, text_chain, all_elements)
        logging.info("Completed batch processing of text elements.")

        # Create page summaries
        logging.info("Creating page summaries.")
        page_prompt_template = PromptTemplate.from_template(
            """
            You are an expert in the AEC industry.
            Your task is to generate a concise summary of the page based on the summaries of its elements.

            Please read the following element summaries and create a page summary that captures the key points and main ideas.

            Element Summaries:
            {element_summaries}

            Generate a summary of the page, highlighting the main points and key information.
            Your summary should be comprehensive and capture the essence of the page in a concise manner.
            """
        )
        logging.debug("Defining page summary chain.")
        page_chain = page_prompt_template | chat_model | StrOutputParser()
        logging.info("Page summary chain defined.")

        # Collect input data for all pages
        logging.debug("Collecting input data for page summaries.")
        page_inputs = []
        page_indices = []
        for idx, page in enumerate(self.metadata.pages):
            element_summaries = "\n".join(
                [
                    element.summary
                    for element in page.elements
                    if hasattr(element, "summary") and element.summary
                ]
            )
            if not element_summaries:
                logging.debug(f"No summaries found for page index {idx}. Skipping.")
                continue  # Skip if there are no summaries
            input_data = {"element_summaries": element_summaries}
            page_inputs.append(input_data)
            page_indices.append(idx)
        logging.info(f"Prepared {len(page_inputs)} inputs for page summaries.")

        # Process page summaries in batch
        logging.info("Starting batch processing of page summaries.")
        page_summaries = page_chain.batch(page_inputs, config={"max_concurrency": 5})
        logging.info("Completed batch processing of page summaries.")

        # Assign summaries back to the respective pages
        logging.debug("Assigning summaries to respective pages.")
        for idx, page_summary in zip(page_indices, page_summaries):
            self.metadata.pages[idx].summary = page_summary
            logging.debug(f"Assigned summary to page index {idx}.")
        logging.info("All page summaries assigned successfully.")

    def get_context_elements(
        self,
        target_id: int,
        all_elements: List,
        window: int = 5,
        min_text_elements: int = 10,
    ):
        element_indices = {
            element.element_id: idx for idx, element in enumerate(all_elements)
        }
        target_idx = element_indices.get(target_id)
        if target_idx is None:
            return []

        current_window = window
        while current_window < len(all_elements):
            start_idx = max(0, target_idx - current_window)
            end_idx = min(len(all_elements), target_idx + current_window + 1)
            context_elements = (
                all_elements[start_idx:target_idx]
                + all_elements[target_idx + 1 : end_idx]
            )
            text_elements = [e for e in context_elements if isinstance(e, TextElement)]
            if len(text_elements) >= min_text_elements:
                break
            current_window += window

        return context_elements

    def convert_image_to_base64(self, image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def batch_process_images(
        self,
        image_elements: List[ImageElement | TableElement],
        all_elements: List,
        llm_model: ChatVertexAI,
        template,
        batch_size=3,
    ):
        logging.info("Starting batch processing of images.")
        image_inputs = []
        for idx, element in enumerate(image_elements, start=1):
            logging.debug(
                f"Processing image {idx}/{len(image_elements)}: {element.path}"
            )
            context_elements = self.get_context_elements(
                element.element_id, all_elements
            )
            context_text = "; ".join(
                [e.text for e in context_elements if isinstance(e, TextElement)]
            )
            try:
                image_base64 = self.convert_image_to_base64(element.path)
                image_data = f"data:image/png;base64,{image_base64}"
                formatted_prompt = template.format(context_text=context_text)
                message = [
                    {"role": "system", "content": formatted_prompt},
                    {"role": "user", "content": image_data},
                ]
                image_inputs.append(message)
            except Exception as e:
                logging.error(f"Error reading image at {element.path}: {e}")
                element.summary = ""

        logging.info(
            f"Sending {len(image_inputs)} images to the image_chain for processing."
        )
        image_descriptions = llm_model.batch(
            image_inputs, config={"max_concurrency": batch_size}
        )

        for idx, (element, description) in enumerate(
            zip(image_elements, image_descriptions), start=1
        ):
            logging.debug(
                f"Assigning summary to image {idx}/{len(image_elements)}: {element.path}"
            )
            element.summary = description.content
        logging.info("Completed batch processing of images.")

    def batch_process_texts(
        self,
        text_elements: List[TextElement],
        text_chain: RunnableSerializable,
        all_elements: List,
        batch_size=5,
    ):
        logging.info("Starting batch processing of text elements.")
        text_inputs = []
        for idx, element in enumerate(text_elements, start=1):
            logging.debug(
                f"Processing text element {idx}/{len(text_elements)}: ID {element.element_id}"
            )
            context_elements = self.get_context_elements(
                element.element_id, all_elements
            )
            context_text = "; ".join(
                [e.text for e in context_elements if isinstance(e, TextElement)]
            )
            input_data = {"context_text": context_text, "text_data": element.text}
            text_inputs.append(input_data)
        logging.info(
            f"Sending {len(text_inputs)} text elements to the text_chain for processing."
        )
        text_summaries = text_chain.batch(
            text_inputs, config={"max_concurrency": batch_size}
        )
        for idx, (element, summary) in enumerate(
            zip(text_elements, text_summaries), start=1
        ):
            logging.debug(
                f"Assigning summary to text element {idx}/{len(text_elements)}: ID {element.element_id}"
            )
            element.summary = summary
        logging.info("Completed batch processing of text elements.")

    def export_data(self):
        result_data = {
            "file_name": self.metadata.file_name,
            "file_size": self.metadata.file_size,
            "page_numbers": self.metadata.page_numbers,
            "pages": [page.model_dump() for page in self.metadata.pages],
        }

        # Define the output file path
        output_dir = "./output"
        output_file_path = os.path.join(output_dir, "result.json")

        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Write the dictionary to a JSON file
        with open(output_file_path, "w") as output_file:
            json.dump(result_data, output_file, indent=4)

from flask import Blueprint, request, jsonify, current_app as app
from service.document_processor import DocumentProcessor
from service.question_answer import Chatbot
from service.vector_store import VectorStore
from service.firebase import FirestoreManager

# Create a Blueprint for modular routes
routes_bp = Blueprint("routes", __name__)


# Initialize services with the app configuration
@routes_bp.before_app_request
def initialize_services():
    global vector_store, document_processor, chatbot
    vector_store = VectorStore(app.config)
    document_processor = DocumentProcessor()
    chatbot = Chatbot(app.config, vector_store)


@routes_bp.route("/get_documents", methods=["GET"])
def get_documents():
    """
    Endpoint to get the list of document metadata.
    """
    try:
        # Initialize FirestoreManager
        firestore_manager = FirestoreManager()

        # Retrieve all document metadata
        documents = firestore_manager.get_all_document_metadata()

        # Return the documents as JSON
        return jsonify(documents), 200
    except Exception as e:
        # Log the error with traceback and custom message
        app.logger.exception(f"Error retrieving documents: {str(e)}")
        return jsonify({"error": str(e)}), 500


@routes_bp.route("/add_document", methods=["POST"])
def add_document():
    """
    Endpoint to upload and process a PDF document.
    """
    try:
        # Check if the "file" part is present in the request
        if "file" not in request.files:
            app.logger.warning("No file part in the request.")
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        # Check if the file name is empty
        if file.filename == "":
            app.logger.warning("No file selected for upload.")
            return jsonify({"error": "No selected file"}), 400

        # Validate the file format
        if file and file.filename.endswith(".pdf"):
            # Process the document using the document processor
            app.logger.info(f"Processing document: {file.filename}")
            result = document_processor.process_document(vector_store, file)

            # Advance document process
            temp_path = f"/tmp/{file.filename}"
            # file.save(temp_path)
            split_document_paths = document_processor.split_pdf_if_needed(temp_path)
            document_processor.parse_document(split_document_paths)
            document_processor.create_summary()
            advance_result = document_processor.advanced_process_document(vector_store)

            if result and advance_result:
                app.logger.info(f"Document {file.filename} processed successfully.")
                return jsonify({"message": "Document added successfully"}), 200
            else:
                error_messages = []
                if not result:
                    error_messages.append("Failed to process document.")
                if not advance_result:
                    error_messages.append("Failed to advance process document.")
                app.logger.error(
                    f"Errors: {', '.join(error_messages)} for document: {file.filename}"
                )
                return jsonify({"error": " ".join(error_messages)}), 500
        else:
            app.logger.warning(f"Invalid file format: {file.filename}")
            return jsonify({"error": "Invalid file format. Please upload a PDF."}), 400
    except Exception as e:
        # Log the error with traceback and custom message
        app.logger.exception(f"Error processing document: {str(e)}")
        return jsonify({"error": str(e)}), 500


@routes_bp.route("/ask", methods=["POST"])
async def ask_question_route():
    """
    Endpoint to ask a question to the chatbot.
    """
    try:
        # Validate incoming JSON payload
        data = request.get_json()
        if not data or "question" not in data:
            app.logger.warning("No question provided in the request.")
            return jsonify({"error": "No question provided"}), 400

        question = data["question"]
        app.logger.info(f"Received question: {question}")

        is_advance_search = False if "advance_search" not in data else True
        app.logger.info(f"Advance search: {is_advance_search}")

        document_guids = data.get("document_guids", [])
        app.logger.info(f"Document GUIDs: {document_guids}")

        # Use the chatbot to get a response
        response = await chatbot.ask_question(
            question, is_advance_search, document_guids
        )
        app.logger.info(f"Chatbot response: {response}")

        return jsonify(response), 200
    except Exception as e:
        # Log the error with traceback and custom message
        app.logger.exception(f"Error while handling the /ask endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

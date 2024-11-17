from flask import Blueprint, request, jsonify, current_app as app
from backend.service.document_processor import DocumentProcessor
from backend.service.question_answer import Chatbot
from backend.service.vector_store import VectorStore

# Create a Blueprint for modular routes
routes_bp = Blueprint("routes", __name__)


# Initialize services with the app configuration
@routes_bp.before_app_request
def initialize_services():
    global vector_store, document_processor, chatbot
    vector_store = VectorStore(app.config)
    document_processor = DocumentProcessor(vector_store)
    chatbot = Chatbot(app.config, vector_store)


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
            result = document_processor.process_document(file)

            if result:
                app.logger.info(f"Document {file.filename} processed successfully.")
                return jsonify({"message": "Document added successfully"}), 200
            else:
                app.logger.error(f"Failed to process document: {file.filename}")
                return jsonify({"error": "Failed to process document"}), 500
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

        # Use the chatbot to get a response
        response = await chatbot.ask_question(question)
        app.logger.info(f"Chatbot response: {response}")

        return jsonify(response), 200
    except Exception as e:
        # Log the error with traceback and custom message
        app.logger.exception(f"Error while handling the /ask endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500


# @routes_bp.route("/get_document_names", methods=["GET"])
# def get_document_names():
#     """
#     Endpoint to get the names of all documents in the vector store.
#     """
#     try:
#         # Get all document names from the vector store
#         document_names = vector_store.get_all_document_names()
#         app.logger.info(f"Document names retrieved: {document_names}")
#         return jsonify(document_names), 200
#     except Exception as e:
#         # Log the error with traceback and custom message
#         app.logger.exception(f"Error getting document names: {str(e)}")
#         return jsonify({"error": str(e)}), 500

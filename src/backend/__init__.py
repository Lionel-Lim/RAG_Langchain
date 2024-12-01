import os
import logging
from flask import Flask
from common.utils import load_yaml


def create_app():
    # Create an instance of the Flask class
    app = Flask(__name__)

    # Debug mode
    app.debug = False  # Set to False in production

    # Load configuration settings
    config_file_path = r"./config.yaml"
    config = load_yaml(config_file_path)

    # Set environment variable for Google Application Credentials
    # if app.debug:
    SERVICE_ACCOUNT_ADDRESS = config["service_account_address"]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_ADDRESS

    # Pass configuration to the app
    app.config.update(config)

    # Register blueprints
    from routes import routes_bp

    app.register_blueprint(routes_bp)

    # Custom error handler
    @app.errorhandler(404)
    def not_found(error):
        app.logger.warning("404 Not Found: %s", error)
        return {"error": "Not found"}, 404

    # Logging configuration
    if not app.debug:  # Only log to file in production
        from logging.handlers import RotatingFileHandler

        # Create a rotating file handler
        file_handler = RotatingFileHandler("error.log", maxBytes=10240, backupCount=10)
        file_handler.setLevel(logging.INFO)

        # Add detailed formatter to include line numbers
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s - [in %(pathname)s:%(lineno)d]"
        )
        file_handler.setFormatter(formatter)

        # Attach the file handler to the Flask logger
        app.logger.addHandler(file_handler)

    # Ensure debug logs include line numbers (works in both debug and production)
    logging.basicConfig(
        level=logging.DEBUG if app.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s - [in %(pathname)s:%(lineno)d]",
        handlers=[logging.StreamHandler()],
    )

    return app

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables for Poetry
ENV POETRY_VERSION=1.8.4 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true 

ENV PATH="$POETRY_HOME/bin:$PATH" 
ENV GOOGLE_CLOUD_PROJECT="ai-sandbox-company-73"

# Google Cloud Project ID
ENV GOOGLE_CLOUD_PROJECT="ai-sandbox-company-73"

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Set the working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies without dev packages
RUN poetry install --no-dev --no-root

# Copy the application code
COPY . .

# Expose the port your app runs on
EXPOSE 4000

# Set environment variables
ENV PYTHONPATH="." \
    FLASK_APP=src/backend/__init__.py:create_app

# Set the PATH to include the virtual environment's bin directory
ENV PATH="/app/.venv/bin:$PATH"

# Run the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:4000", "__init__:create_app()"]
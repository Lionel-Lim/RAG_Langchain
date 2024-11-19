# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables for Poetry
ENV POETRY_VERSION=1.8.4 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_HOME="/opt/poetry"

ENV GOOGLE_CLOUD_PROJECT="ai-sandbox-company-73"

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Set the working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install all dependencies globally
RUN pip install aiohttp==3.11.5
RUN poetry config virtualenvs.create false && poetry install --no-dev
# RUN poetry install --only main

# Copy the rest of the application code
COPY . .

# Expose the port your app runs on
EXPOSE 4000

# Set environment variables
ENV PYTHONPATH="." \
    FLASK_APP=src/backend/__init__.py:create_app

# Run the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:4000", "__init__:create_app()"]
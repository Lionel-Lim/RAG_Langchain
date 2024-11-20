# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables for Poetry
ENV POETRY_VERSION=1.8.4 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PATH="$POETRY_HOME/bin:$PATH"

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
EXPOSE 8501

# Run the frontend application (replace 'app.py' with your entry point)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
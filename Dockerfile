FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies for compiling Python packages (like hdbscan)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install uv

# Copy dependency files first to maximize Docker layer caching
COPY pyproject.toml uv.lock ./

# Sync dependencies using uv (creates a .venv automatically)
RUN uv sync

# Copy the rest of the application code
COPY . .

# Expose the API port
EXPOSE 8000

CMD ["sh", "-c", "exec uv run uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

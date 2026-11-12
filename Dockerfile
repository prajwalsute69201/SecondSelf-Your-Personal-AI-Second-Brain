# SecondSelf — Production Dockerfile with Persistent Storage
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SECONDSELF_ROOT=/app \
    SECONDSELF_DATA_DIR=/app/data \
    SECONDSELF_DB_PATH=/app/data/secondself.db

WORKDIR /app

# Install minimal build and runtime tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache sentence-transformers embedding model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application source code
COPY . /app

# Ensure directories exist
RUN mkdir -p /app/data /app/wiki/Projects /app/wiki/Areas /app/wiki/Resources /app/wiki/Archives /app/raw /app/backups

# Declare persistent volumes for durable storage across container restarts & redeploys
VOLUME ["/app/data", "/app/wiki", "/app/raw", "/app/backups"]

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


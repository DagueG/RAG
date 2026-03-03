# Multi-stage build for RAG API
# Stage 1: Builder - Download models and build index (if not pre-built)
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy project files for dependency installation
COPY pyproject.toml setup.py* uv.lock* ./

# Use uv to install dependencies
RUN uv pip install --system -e .

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# Build index if not pre-built (optional - index should be pre-built)
RUN if [ ! -f ./data/faiss_index/faiss_index.bin ]; then \
    python src/data_processing/fetch_events.py && \
    python src/data_processing/clean_data.py && \
    python src/vectorization/build_index.py; \
    fi

# Stage 2: Runtime - Lean production image
FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies (no build tools)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and pre-built data
COPY --from=builder /app/src ./src
COPY --from=builder /app/data ./data
COPY pyproject.toml setup.py* ./

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

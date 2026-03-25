FROM python:3.11-slim

# Hugging Face Spaces runs containers as UID 1000; set up a matching non-root user
# with a writable home so model caches/lockfiles work at runtime. This is also
# harmless on Cloud Run / local Docker.
RUN useradd -m -u 1000 user

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER user

# Pre-bake all models into the (user-owned) cache for fast, offline-capable cold starts.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" && \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" && \
    python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25')"

COPY --chown=user app ./app
COPY --chown=user scripts ./scripts
COPY --chown=user sample_docs ./sample_docs
COPY --chown=user eval ./eval
COPY --chown=user frontend ./frontend

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

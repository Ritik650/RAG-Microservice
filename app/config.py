"""Central configuration, loaded from environment / .env.

Field names map to env vars case-insensitively, e.g. ``qdrant_url`` <- ``QDRANT_URL``
and ``gemini_api_key`` <- ``GEMINI_API_KEY`` (the name the google-genai SDK also reads).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Generation (Gemini) ---
    gemini_api_key: str | None = None
    llm_model: str = "gemini-2.5-flash"  # free tier; fast, streams well for RAG answers
    llm_max_tokens: int = 1024

    # --- Vector store (Qdrant) ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "rag_chunks"

    # --- Embeddings (sentence-transformers) ---
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384  # must match embed_model's output dimension

    # --- Chunking / retrieval ---
    chunk_size: int = 800      # characters per chunk (word-boundary aware)
    chunk_overlap: int = 120   # characters of overlap between adjacent chunks
    top_k: int = 5             # default number of chunks retrieved per query

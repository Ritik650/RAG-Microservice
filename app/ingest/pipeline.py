"""Async ingestion pipeline: chunk -> embed -> upsert. Idempotent by design."""

from __future__ import annotations

import uuid
from collections.abc import Iterable

from fastapi.concurrency import run_in_threadpool

from app.ingest.chunk import chunk_text

# Fixed namespace so point ids are stable across runs -> re-ingesting the same
# document overwrites its chunks instead of creating duplicates.
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


def point_id(source: str, index: int) -> str:
    """Deterministic UUIDv5 for a (source, chunk_index) pair."""
    return str(uuid.uuid5(_NAMESPACE, f"{source}::{index}"))


async def ingest_documents(documents: Iterable, embedder, sparse_encoder, store, settings) -> dict:
    records: list[dict] = []
    doc_count = 0
    for doc in documents:
        doc_count += 1
        chunks = chunk_text(doc.text, settings.chunk_size, settings.chunk_overlap)
        for i, chunk in enumerate(chunks):
            records.append(
                {
                    "id": point_id(doc.source, i),
                    "source": doc.source,
                    "chunk_index": i,
                    "text": chunk,
                }
            )

    if not records:
        return {"documents": doc_count, "chunks_upserted": 0}

    await store.ensure_ready(settings.embed_dim)
    # Embedding is CPU-bound and synchronous -> run off the event loop.
    texts = [r["text"] for r in records]
    vectors = await run_in_threadpool(embedder.encode, texts)
    sparse = await run_in_threadpool(sparse_encoder.encode_docs, texts)
    points = [
        {**r, "vector": v, "sparse_indices": idx, "sparse_values": val}
        for r, v, (idx, val) in zip(records, vectors, sparse, strict=True)
    ]
    await store.upsert(points)
    return {"documents": doc_count, "chunks_upserted": len(points)}

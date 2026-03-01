"""Dense (embedding) retrieval. Hybrid BM25 + RRF fusion lands in Week 2."""

from __future__ import annotations

from fastapi.concurrency import run_in_threadpool


async def dense_search(question: str, embedder, store, top_k: int) -> list[dict]:
    vector = (await run_in_threadpool(embedder.encode, [question]))[0]
    return await store.search(vector, top_k)

"""Dense-only retrieval. Kept as a baseline for before/after hybrid+rerank evals."""

from __future__ import annotations

from fastapi.concurrency import run_in_threadpool


async def dense_search(question: str, embedder, store, top_k: int) -> list[dict]:
    vector = (await run_in_threadpool(embedder.encode, [question]))[0]
    return await store.search_dense(vector, top_k)

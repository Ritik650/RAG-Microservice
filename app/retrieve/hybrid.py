"""Hybrid retrieval: dense + BM25 sparse searched in parallel, fused with RRF."""

from __future__ import annotations

import asyncio

from fastapi.concurrency import run_in_threadpool

from app.retrieve.fusion import rrf


async def hybrid_search(
    question: str,
    embedder,
    sparse_encoder,
    store,
    limit: int,
    rrf_k: int = 60,
) -> list[dict]:
    """Return up to ``limit`` fused candidates. ``score`` is the RRF score."""
    dense_vec = (await run_in_threadpool(embedder.encode, [question]))[0]
    indices, values = await run_in_threadpool(sparse_encoder.encode_query, question)

    dense_hits, sparse_hits = await asyncio.gather(
        store.search_dense(dense_vec, limit),
        store.search_sparse(indices, values, limit),
    )

    fused = rrf(
        [[h["id"] for h in dense_hits], [h["id"] for h in sparse_hits]],
        k=rrf_k,
    )
    by_id = {h["id"]: h for h in dense_hits + sparse_hits}
    return [{**by_id[doc_id], "score": score} for doc_id, score in fused[:limit]]

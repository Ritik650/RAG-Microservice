"""Retrieval evaluation runner.

Evaluates three retrieval configurations against the labeled QA set so the
dense → hybrid → hybrid+rerank quality progression is measurable:

- "dense":         dense vectors only
- "hybrid":        dense + BM25 fused with RRF
- "hybrid_rerank": hybrid candidates re-scored by the cross-encoder

For each mode it reports recall@k, MRR, nDCG@k, and p50/p95 latency (ms).
LLM-free: only retrieval runs, so it is cheap enough for CI.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.concurrency import run_in_threadpool

from app.evaluation.metrics import aggregate, ndcg_at_k, recall_at_k, reciprocal_rank
from app.retrieve.dense import dense_search
from app.retrieve.hybrid import hybrid_search

MODES = ("dense", "hybrid", "hybrid_rerank")


def load_qa_set(path: str | Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct), len(ordered) - 1)
    return round(ordered[idx], 1)


async def _retrieve(mode: str, question: str, deps, settings) -> list[dict]:
    embedder, sparse_encoder, store, reranker = deps
    if mode == "dense":
        return await dense_search(question, embedder, store, settings.top_k)
    candidates = await hybrid_search(
        question,
        embedder,
        sparse_encoder,
        store,
        limit=max(settings.candidate_k, settings.top_k),
        rrf_k=settings.rrf_k,
    )
    if mode == "hybrid_rerank":
        return await run_in_threadpool(reranker.rerank, question, candidates, settings.top_k)
    return candidates[: settings.top_k]


async def run_retrieval_eval(
    qa_items: list[dict], embedder, sparse_encoder, store, reranker, settings
) -> dict:
    """Return {"k": top_k, "questions": n, "modes": {mode: {metrics..., latency...}}}."""
    deps = (embedder, sparse_encoder, store, reranker)
    k = settings.top_k
    results: dict = {"k": k, "questions": len(qa_items), "modes": {}}

    for mode in MODES:
        per_query: list[dict[str, float]] = []
        latencies: list[float] = []
        for item in qa_items:
            expected = set(item["sources"])
            start = time.perf_counter()
            hits = await _retrieve(mode, item["question"], deps, settings)
            latencies.append((time.perf_counter() - start) * 1000)
            retrieved = [h["source"] for h in hits]
            per_query.append(
                {
                    f"recall@{k}": recall_at_k(retrieved, expected, k),
                    "mrr": reciprocal_rank(retrieved, expected),
                    f"ndcg@{k}": ndcg_at_k(retrieved, expected, k),
                }
            )
        results["modes"][mode] = {
            **aggregate(per_query),
            "p50_ms": _percentile(latencies, 0.50),
            "p95_ms": _percentile(latencies, 0.95),
        }
    return results

"""Reciprocal Rank Fusion: combine ranked lists using ranks, not raw scores.

Pure and deterministic, so retrievers whose scores live on different scales
(cosine similarity vs. BM25) can be fused without any normalization.
"""

from __future__ import annotations

from collections.abc import Sequence


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id lists. Each id scores sum(1 / (k + rank_i + 1)) over the
    lists it appears in (rank is 0-based). Returns (id, score) sorted by score
    descending, with id as a deterministic tie-break.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

"""Pure retrieval metrics: recall@k, MRR, nDCG@k.

All functions take the ranked list of retrieved source labels (one per chunk, in
rank order) and the set of expected/relevant source labels. Relevance is binary:
a retrieved chunk is relevant iff its source is in ``expected``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], expected: set[str], k: int) -> float:
    """Fraction of expected sources that appear in the top-k retrieved chunks."""
    if not expected:
        return 0.0
    found = {src for src in retrieved[:k] if src in expected}
    return len(found) / len(expected)


def reciprocal_rank(retrieved: Sequence[str], expected: set[str]) -> float:
    """1 / rank of the first relevant chunk (0.0 if none retrieved)."""
    for i, src in enumerate(retrieved):
        if src in expected:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], expected: set[str], k: int) -> float:
    """Binary source-level nDCG@k: each expected source is credited once, at the
    rank of its first retrieved chunk. Ideal DCG places every expected source at
    the top, so the score is always in [0, 1]."""
    if not expected:
        return 0.0
    dcg = 0.0
    seen: set[str] = set()
    for i, src in enumerate(retrieved[:k]):
        if src in expected and src not in seen:
            dcg += 1.0 / math.log2(i + 2)
            seen.add(src)
    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of per-query metric dicts into one dict."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {key: round(sum(m[key] for m in per_query) / len(per_query), 4) for key in keys}

"""Sparse BM25 embeddings via fastembed's Qdrant/bm25 model.

Documents get BM25 term-frequency weights; queries get plain term indices, and
Qdrant applies IDF server-side (the collection's sparse vectors use Modifier.IDF).
Loaded lazily so importing the app (and running tests) does not require fastembed.
"""

from __future__ import annotations


class SparseEncoder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import SparseTextEmbedding

            self._model = SparseTextEmbedding(self.model_name)
        return self._model

    def encode_docs(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """BM25 doc-side weights as (indices, values) pairs. Synchronous/CPU-bound."""
        return [(e.indices.tolist(), e.values.tolist()) for e in self._load().embed(texts)]

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        """Query-side term indices (IDF weighting happens in Qdrant)."""
        e = next(iter(self._load().query_embed(text)))
        return e.indices.tolist(), e.values.tolist()

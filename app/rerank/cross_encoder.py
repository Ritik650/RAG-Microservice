"""Cross-encoder reranking: jointly score (query, passage) pairs for precision.

Applied only to the small fused candidate set, so the extra cost is bounded.
The model loads lazily so importing the app (and tests) does not pull in torch.
"""

from __future__ import annotations


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, question: str, hits: list[dict], top_k: int) -> list[dict]:
        """Re-score hits with the cross-encoder and return the top_k best.

        Synchronous/CPU-bound — call via run_in_threadpool. ``score`` on the
        returned hits is the cross-encoder relevance score.
        """
        if not hits:
            return []
        scores = self._load().predict([(question, h["text"]) for h in hits])
        ranked = sorted(zip(hits, scores, strict=True), key=lambda pair: -float(pair[1]))
        return [{**h, "score": float(s)} for h, s in ranked[:top_k]]

"""Dense embeddings via sentence-transformers.

The heavy model is loaded lazily on first ``encode`` so that importing the app
(and running unit tests) does not pull in torch.
"""

from __future__ import annotations


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts into L2-normalized vectors (cosine-ready). Synchronous/CPU-bound."""
        model = self._load()
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [v.tolist() for v in vectors]

    @property
    def dim(self) -> int:
        return self._load().get_sentence_embedding_dimension()

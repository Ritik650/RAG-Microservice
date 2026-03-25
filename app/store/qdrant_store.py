"""Async Qdrant wrapper: hybrid collection (dense + BM25 sparse), idempotent upsert.

The collection holds two named vectors per point:
- "dense": cosine sentence-transformer embeddings
- "bm25":  sparse BM25 term weights with server-side IDF (Modifier.IDF)
"""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient, models

from app.config import Settings

DENSE = "dense"
SPARSE = "bm25"


class QdrantStore:
    def __init__(self, settings: Settings) -> None:
        self.collection = settings.collection_name
        # check_compatibility=False keeps construction offline (no version probe on init).
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            check_compatibility=False,
        )
        self._ready = False

    async def ensure_ready(self, dim: int) -> None:
        """Create the hybrid collection on first use if it doesn't exist yet."""
        if self._ready:
            return
        if not await self.client.collection_exists(self.collection):
            await self.client.create_collection(
                self.collection,
                vectors_config={
                    DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
        self._ready = True

    async def upsert(self, points: list[dict]) -> None:
        """Upsert points. Deterministic ids (see pipeline.point_id) make this idempotent."""
        structs = [
            models.PointStruct(
                id=p["id"],
                vector={
                    DENSE: p["vector"],
                    SPARSE: models.SparseVector(
                        indices=p["sparse_indices"], values=p["sparse_values"]
                    ),
                },
                payload={
                    "source": p["source"],
                    "chunk_index": p["chunk_index"],
                    "text": p["text"],
                },
            )
            for p in points
        ]
        await self.client.upsert(self.collection, points=structs, wait=True)

    async def search_dense(self, vector: list[float], top_k: int) -> list[dict]:
        res = await self.client.query_points(
            self.collection, query=vector, using=DENSE, limit=top_k, with_payload=True
        )
        return [self._hit(p) for p in res.points]

    async def search_sparse(
        self, indices: list[int], values: list[float], top_k: int
    ) -> list[dict]:
        res = await self.client.query_points(
            self.collection,
            query=models.SparseVector(indices=indices, values=values),
            using=SPARSE,
            limit=top_k,
            with_payload=True,
        )
        return [self._hit(p) for p in res.points]

    @staticmethod
    def _hit(p) -> dict:
        payload = p.payload or {}
        return {
            "id": str(p.id),
            "score": p.score,
            "source": payload.get("source", ""),
            "chunk_index": payload.get("chunk_index", 0),
            "text": payload.get("text", ""),
        }

    async def count(self) -> int:
        """Number of indexed chunks, or -1 if the collection hasn't been created yet
        (distinct from a connectivity failure, which propagates as an exception)."""
        if not await self.client.collection_exists(self.collection):
            return -1
        return (await self.client.count(self.collection, exact=True)).count

    async def close(self) -> None:
        await self.client.close()

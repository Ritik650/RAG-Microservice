"""Async Qdrant wrapper: collection lifecycle, idempotent upsert, dense search."""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient, models

from app.config import Settings


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
        """Create the collection on first use if it doesn't exist yet."""
        if self._ready:
            return
        if not await self.client.collection_exists(self.collection):
            await self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
        self._ready = True

    async def upsert(self, points: list[dict]) -> None:
        """Upsert points. Deterministic ids (see pipeline.point_id) make this idempotent."""
        structs = [
            models.PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload={
                    "source": p["source"],
                    "chunk_index": p["chunk_index"],
                    "text": p["text"],
                },
            )
            for p in points
        ]
        await self.client.upsert(self.collection, points=structs, wait=True)

    async def search(self, vector: list[float], top_k: int) -> list[dict]:
        res = await self.client.query_points(
            self.collection, query=vector, limit=top_k, with_payload=True
        )
        out: list[dict] = []
        for p in res.points:
            payload = p.payload or {}
            out.append(
                {
                    "id": str(p.id),
                    "score": p.score,
                    "source": payload.get("source", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "text": payload.get("text", ""),
                }
            )
        return out

    async def count(self) -> int:
        return (await self.client.count(self.collection, exact=True)).count

    async def close(self) -> None:
        await self.client.close()

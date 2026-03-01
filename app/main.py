"""FastAPI entrypoint: /healthz, /ingest, /query (JSON or SSE streaming)."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse

from app.config import Settings
from app.embed.embedder import Embedder
from app.generate.llm import GeminiGenerator
from app.ingest.pipeline import ingest_documents
from app.models import (
    Citation,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.retrieve.dense import dense_search
from app.store.qdrant_store import QdrantStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings
    # All constructed cheaply: the embedding model and Gemini client load lazily,
    # and the Qdrant client does not connect until first use.
    app.state.embedder = Embedder(settings.embed_model)
    app.state.store = QdrantStore(settings)
    app.state.generator = GeminiGenerator(settings)
    try:
        yield
    finally:
        await app.state.store.close()


app = FastAPI(title="RAG Retrieval Microservice", version="0.1.0", lifespan=lifespan)


# --- Dependencies (overridable in tests) ---
def get_settings() -> Settings:
    return app.state.settings


def get_embedder() -> Embedder:
    return app.state.embedder


def get_store() -> QdrantStore:
    return app.state.store


def get_generator() -> GeminiGenerator:
    return app.state.generator


def _citation(c: dict) -> Citation:
    return Citation(
        id=c["id"],
        source=c["source"],
        chunk_index=c["chunk_index"],
        score=round(float(c["score"]), 4),
        snippet=c["text"][:240],
    )


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/healthz")
async def healthz(settings: Settings = Depends(get_settings), store: QdrantStore = Depends(get_store)):
    out = {
        "status": "ok",
        "collection": settings.collection_name,
        "llm_model": settings.llm_model,
        "embed_model": settings.embed_model,
    }
    try:
        out["chunks_indexed"] = await store.count()
        out["qdrant"] = "reachable"
    except Exception:
        out["qdrant"] = "unreachable"
    return out


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    req: IngestRequest,
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    store: QdrantStore = Depends(get_store),
):
    result = await ingest_documents(req.documents, embedder, store, settings)
    return IngestResponse(
        documents=result["documents"],
        chunks_upserted=result["chunks_upserted"],
        collection=settings.collection_name,
    )


@app.post("/query")
async def query(
    req: QueryRequest,
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    store: QdrantStore = Depends(get_store),
    generator: GeminiGenerator = Depends(get_generator),
):
    await store.ensure_ready(settings.embed_dim)
    top_k = req.top_k or settings.top_k
    contexts = await dense_search(req.question, embedder, store, top_k)
    citations = [_citation(c) for c in contexts]

    if req.stream:
        async def event_stream():
            async for token in generator.stream(req.question, contexts):
                yield _sse_event("token", {"text": token})
            yield _sse_event("citations", {"citations": [c.model_dump() for c in citations]})
            yield _sse_event("done", {})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    answer = await generator.generate(req.question, contexts)
    return QueryResponse(answer=answer, citations=citations)

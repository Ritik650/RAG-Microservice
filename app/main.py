"""FastAPI entrypoint: /healthz, /ingest, /query (JSON or SSE streaming), /eval,
plus the static chat UI served from frontend/ when present."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.auth.jwt_auth import require_auth
from app.config import Settings
from app.embed.embedder import Embedder
from app.embed.sparse import SparseEncoder
from app.evaluation.runner import load_qa_set, run_retrieval_eval
from app.generate.llm import GeminiGenerator
from app.ingest.parsers import parse_bytes
from app.ingest.pipeline import ingest_documents
from app.models import (
    Citation,
    Document,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.rerank.cross_encoder import CrossEncoderReranker
from app.retrieve.hybrid import hybrid_search
from app.store.qdrant_store import QdrantStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings
    # All constructed cheaply: models and the Gemini client load lazily,
    # and the Qdrant client does not connect until first use.
    app.state.embedder = Embedder(settings.embed_model)
    app.state.sparse_encoder = SparseEncoder(settings.sparse_model)
    app.state.store = QdrantStore(settings)
    app.state.generator = GeminiGenerator(settings)
    app.state.reranker = CrossEncoderReranker(settings.rerank_model)
    try:
        yield
    finally:
        await app.state.store.close()


app = FastAPI(title="RAG Retrieval Microservice", version="0.1.0", lifespan=lifespan)

# Allow a separately-hosted chat UI (e.g. Vercel) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Dependencies (overridable in tests) ---
def get_settings() -> Settings:
    return app.state.settings


def get_embedder() -> Embedder:
    return app.state.embedder


def get_sparse_encoder() -> SparseEncoder:
    return app.state.sparse_encoder


def get_reranker() -> CrossEncoderReranker:
    return app.state.reranker


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
async def healthz(
    settings: Settings = Depends(get_settings), store: QdrantStore = Depends(get_store)
):
    out = {
        "status": "ok",
        "collection": settings.collection_name,
        "llm_model": settings.llm_model,
        "embed_model": settings.embed_model,
        "sparse_model": settings.sparse_model,
        "rerank_model": settings.rerank_model if settings.rerank_enabled else None,
    }
    try:
        count = await store.count()
        out["qdrant"] = "reachable"
        # -1 means the collection hasn't been created yet (fresh cluster, no ingest).
        out["chunks_indexed"] = count if count >= 0 else 0
        out["collection_ready"] = count >= 0
    except Exception:
        out["qdrant"] = "unreachable"
    return out


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    req: IngestRequest,
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    sparse_encoder: SparseEncoder = Depends(get_sparse_encoder),
    store: QdrantStore = Depends(get_store),
    _claims: dict | None = Depends(require_auth),
):
    result = await ingest_documents(req.documents, embedder, sparse_encoder, store, settings)
    return IngestResponse(
        documents=result["documents"],
        chunks_upserted=result["chunks_upserted"],
        collection=settings.collection_name,
    )


@app.post("/ingest/files", response_model=IngestResponse)
async def ingest_files(
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    sparse_encoder: SparseEncoder = Depends(get_sparse_encoder),
    store: QdrantStore = Depends(get_store),
    _claims: dict | None = Depends(require_auth),
):
    """Upload documents directly (multipart): PDF, Word (.docx), HTML, CSV/TSV,
    txt/md/rst/log. Parsed to text, then chunked/embedded/upserted like /ingest."""
    documents = []
    for file in files:
        name = file.filename or "upload"
        data = await file.read()
        try:
            # Parsing (esp. PDF/DOCX) is CPU-bound -> off the event loop.
            text = await run_in_threadpool(parse_bytes, name, data)
        except ValueError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Could not parse {name!r}: {exc}"
            ) from exc
        if text.strip():
            documents.append(Document(source=name, text=text))

    result = await ingest_documents(documents, embedder, sparse_encoder, store, settings)
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
    sparse_encoder: SparseEncoder = Depends(get_sparse_encoder),
    store: QdrantStore = Depends(get_store),
    generator: GeminiGenerator = Depends(get_generator),
    reranker: CrossEncoderReranker = Depends(get_reranker),
    _claims: dict | None = Depends(require_auth),
):
    await store.ensure_ready(settings.embed_dim)
    top_k = req.top_k or settings.top_k

    # Hybrid retrieve a wider candidate set, then rerank down to top_k.
    candidates = await hybrid_search(
        req.question,
        embedder,
        sparse_encoder,
        store,
        limit=max(settings.candidate_k, top_k),
        rrf_k=settings.rrf_k,
    )
    if settings.rerank_enabled:
        contexts = await run_in_threadpool(reranker.rerank, req.question, candidates, top_k)
    else:
        contexts = candidates[:top_k]
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


@app.post("/eval")
async def evaluate(
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    sparse_encoder: SparseEncoder = Depends(get_sparse_encoder),
    store: QdrantStore = Depends(get_store),
    reranker: CrossEncoderReranker = Depends(get_reranker),
):
    """Run the LLM-free retrieval eval (dense vs hybrid vs hybrid+rerank) on the
    labeled QA set against the live index."""
    qa_path = Path(settings.qa_set_path)
    if not qa_path.exists():
        raise HTTPException(status_code=404, detail=f"QA set not found at {qa_path}")
    qa_items = load_qa_set(qa_path)
    await store.ensure_ready(settings.embed_dim)
    return await run_retrieval_eval(qa_items, embedder, sparse_encoder, store, reranker, settings)


# Serve the chat UI at / when the frontend directory is present (mounted last so
# API routes above take precedence).
_frontend = Path(__file__).resolve().parent.parent / "frontend"
if _frontend.is_dir():
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")

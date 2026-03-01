# RAG Retrieval Microservice with Evaluation

A corpus-agnostic document Q&A service — drop in a folder of docs, it ingests them,
and answers questions with **inline citations** and **token streaming**. The
differentiator (landing in Week 3) is a **retrieval-quality evaluation harness wired
into CI as a regression test**: retrieval quality is measured against a hand-labeled
QA set, and the build fails if scores drop below a threshold.

This repository currently implements the **Week-1 vertical slice**: an end-to-end,
runnable core.

```
                Frontend chat UI (later)
                        │
                        ▼
        ┌──────────────────────────────────┐
        │  FastAPI service (app/main.py)    │
        │   ├── POST /ingest   chunk→embed→upsert (async, idempotent)
        │   ├── POST /query    dense retrieve → generate (JSON or SSE stream + citations)
        │   └── GET  /healthz  status + indexed chunk count
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Qdrant (vector DB, cosine)       │  ← dense sentence-transformers vectors
        └──────────────┬───────────────────┘
                       ▼
        Google Gemini (gemini-2.5-flash) — grounded answer + [n] citations
```

## Stack

| Piece | Choice |
|-------|--------|
| API | FastAPI + Uvicorn (async) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine) |
| Vector store | Qdrant (self-hosted via docker-compose; Qdrant Cloud free tier in prod) |
| Generation | Google Gemini `gemini-2.5-flash` (free tier), streamed |
| Config | `pydantic-settings` (env / `.env`) |

## Quickstart (Docker)

```bash
cp .env.example .env          # add your GEMINI_API_KEY
docker compose up --build     # starts Qdrant + the app on :8000
```

Ingest the bundled sample corpus, then ask a question:

```bash
python scripts/load_folder.py sample_docs

curl -s localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question": "What is Reciprocal Rank Fusion?"}' | jq
```

Stream tokens as Server-Sent Events with inline citations:

```bash
curl -N localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question": "Why should a RAG system cite sources?", "stream": true}'
```

## API

| Endpoint | Description |
|----------|-------------|
| `GET /healthz` | Status, configured models, and count of indexed chunks. |
| `POST /ingest` | Body `{"documents": [{"source": "...", "text": "..."}]}`. Chunks → embeds → upserts. Idempotent. |
| `POST /query` | Body `{"question": "...", "top_k": 5, "stream": false}`. Returns `{answer, citations[]}`, or an SSE token stream when `stream: true`. |

## Design notes

- **Async ingestion, idempotent.** Documents are chunked (word-boundary aware, with
  overlap), embedded off the event loop, and upserted. Point ids are deterministic
  UUIDv5s of `source::chunk_index`, so re-ingesting a document overwrites its chunks
  instead of duplicating them.
- **Corpus-agnostic.** `scripts/load_folder.py` ingests any folder of `.txt` / `.md`
  / `.pdf`. The `/ingest` API itself is plain text in, so any source works.
- **Citations done right.** Every answer references the chunks it used as `[n]`, and
  `/query` returns the corresponding `{id, source, chunk_index, score, snippet}`.
- **Streaming.** `/query` with `stream: true` emits SSE `token` events, then a
  `citations` event, then `done`.

## Development

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on *nix
pip install -r requirements-dev.txt
pytest
```

Unit tests cover the chunker and idempotent id derivation; an integration test drives
`/ingest` → `/query` (JSON and streaming) with fakes, so tests need no torch, no
Qdrant, and no API key.

## Roadmap

- **Week 1 (done):** ingestion + dense retrieval + grounded generation + citations +
  streaming, containerized with Qdrant.
- **Week 2:** BM25 sparse retrieval + RRF hybrid fusion, cross-encoder reranking,
  chat UI.
- **Week 3:** hand-labeled `qa_set.jsonl`, RAGAS eval harness (`/eval`), and a GitHub
  Actions gate that fails the build when retrieval quality regresses — plus a
  before/after-reranking metrics table in this README.

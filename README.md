---
title: RAG Microservice
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# RAG Retrieval Microservice with Evaluation

A corpus-agnostic document Q&A service — upload or drop in a folder of documents
(**PDF, Word, HTML, CSV, markdown, plain text**), it ingests them, and answers
questions with **inline citations** and **token streaming** through a built-in
chat UI. The differentiator: a **retrieval-quality evaluation harness wired
into CI as a regression test**. Retrieval is measured against a hand-labeled QA set
(recall@k, MRR, nDCG — dense vs. hybrid vs. hybrid+rerank), and **the build fails
if scores drop below threshold**.

```
        Chat UI (frontend/, served at /) — SSE streaming + citation cards
                        │
                        ▼
        ┌────────────────────────────────────────────┐
        │  FastAPI service (app/main.py)              │
        │   ├── POST /ingest        chunk→embed (dense+BM25)→upsert (async, idempotent)
        │   ├── POST /ingest/files  multipart upload: PDF/DOCX/HTML/CSV/text → same pipeline
        │   ├── POST /query   hybrid retrieve → RRF → rerank → generate
        │   │                 (JSON or SSE stream, inline citations)
        │   ├── POST /eval    retrieval eval on the labeled QA set (LLM-free)
        │   └── GET  /healthz status + indexed chunk count
        │   (optional JWT bearer auth on /ingest and /query)
        └──────────────┬─────────────────────────────┘
                       ▼
        ┌────────────────────────────────────────────┐
        │  Qdrant — one collection, two named vectors │
        │   ├── "dense": MiniLM cosine embeddings     │
        │   └── "bm25":  sparse BM25, server-side IDF │
        └──────────────┬─────────────────────────────┘
                       ▼
          RRF fusion (client-side, unit-tested)
                       ▼
          Cross-encoder reranker (top-20 → top-5)
                       ▼
        Google Gemini (gemini-2.5-flash) — grounded answer + [n] citations
```

## Stack

| Piece | Choice |
|-------|--------|
| API | FastAPI + Uvicorn (async) |
| Dense embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine) |
| Sparse embeddings | fastembed `Qdrant/bm25` (BM25 term weights, IDF applied by Qdrant) |
| Fusion | Reciprocal Rank Fusion, client-side ([fusion.py](app/retrieve/fusion.py)) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` on the fused top-20 |
| Vector store | Qdrant (self-hosted via docker-compose; Qdrant Cloud free tier in prod) |
| Generation | Google Gemini `gemini-2.5-flash` (free tier), streamed |
| Frontend | Static chat UI (Tailwind), SSE streaming, citation cards |
| Auth | Optional JWT bearer (HS256, PyJWT) |
| Eval | recall@k / MRR / nDCG harness + RAGAS, gated in GitHub Actions |

## Quickstart (Docker)

```bash
cp .env.example .env          # add your GEMINI_API_KEY
docker compose up --build     # starts Qdrant + the app on :8000
```

Ingest the bundled sample corpus, then open **http://localhost:8000** for the chat
UI, or use the API directly:

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
| `GET /healthz` | Status, configured models, count of indexed chunks. |
| `POST /ingest` | Body `{"documents": [{"source": "...", "text": "..."}]}`. Chunks → embeds (dense + BM25) → upserts. Idempotent. |
| `POST /ingest/files` | Multipart file upload. Parses `.pdf` (pypdf), `.docx` (python-docx, incl. tables), `.html/.htm` (tag-stripping), `.csv/.tsv`, `.txt/.md/.rst/.log`, then runs the same pipeline. 415 on unsupported types. |
| `POST /query` | Body `{"question": "...", "top_k": 5, "stream": false}`. Hybrid retrieve → RRF → rerank → generate. Returns `{answer, citations[]}` or an SSE stream. |
| `POST /eval` | Runs the LLM-free retrieval eval (dense vs hybrid vs hybrid_rerank) on [eval/qa_set.jsonl](eval/qa_set.jsonl) against the live index. |
| `GET /` | The chat UI. |

## The evaluation harness (the point of this project)

[eval/qa_set.jsonl](eval/qa_set.jsonl) holds 22 hand-labeled
question → ground-truth → source triples over the sample corpus. The harness
retrieves with three configurations and scores each with **recall@k, MRR, nDCG@k,
and p50/p95 latency**, making the dense → hybrid → reranked quality progression
measurable:

```bash
docker compose up -d qdrant
pip install -r requirements.txt
python eval/run_ragas.py            # ingests sample_docs, prints the table, gates
```

Measured on the sample corpus (22 questions, k=5, local CPU):

| mode | recall@5 | MRR | nDCG@5 | p50 ms | p95 ms |
|------|---------:|----:|-------:|-------:|-------:|
| dense | 1.0 | 0.9545 | 0.9664 | 25.7 | 30.6 |
| hybrid (RRF) | 1.0 | 0.9545 | 0.9664 | 31.3 | 33.4 |
| **hybrid + rerank** | **1.0** | **1.0** | **1.0** | 339.7 | 432.9 |

The reranker buys perfect first-hit ranking (MRR 0.9545 → 1.0) for ~300 ms of
CPU-side latency — exactly the precision/latency trade-off the eval makes visible.

The script **exits non-zero** if `hybrid_rerank` falls below `EVAL_MIN_RECALL`
(default 0.85) or `EVAL_MIN_MRR` (default 0.70). CI runs exactly this
([.github/workflows/ci.yml](.github/workflows/ci.yml)):
**lint → tests → retrieval eval gate → docker build** — a retrieval regression
fails the build like a failing unit test. With a `GEMINI_API_KEY` secret
configured, the gate also runs **RAGAS** (faithfulness, answer relevancy, context
precision/recall) via `--ragas` and gates on faithfulness.

Swap the corpus by pointing `--corpus` at any folder and relabeling
`qa_set.jsonl` — the harness is corpus-agnostic like the service.

## Auth (optional)

Set `AUTH_ENABLED=true` and a strong `JWT_SECRET`; `/ingest` and `/query` then
require `Authorization: Bearer <HS256 JWT>`. Mint a demo token:

```bash
python scripts/make_token.py demo-user
```

The chat UI sends the token from `localStorage.RAG_TOKEN` automatically.

## Design notes

- **Async ingestion, idempotent.** Documents are chunked (word-boundary aware, with
  overlap), embedded off the event loop (dense + BM25 sparse in one pass), and
  upserted. Point ids are deterministic UUIDv5s of `source::chunk_index`, so
  re-ingesting a document overwrites its chunks instead of duplicating them.
- **Any document type.** One parser registry ([parsers.py](app/ingest/parsers.py))
  serves both the `/ingest/files` upload endpoint and `scripts/load_folder.py`:
  PDF, Word (paragraphs + tables), HTML, CSV/TSV, and plain-text formats. Adding a
  format is one function + one dict entry. The chat UI's 📎 button uploads directly.
- **Hybrid search.** Every query runs dense (semantic) and BM25 (lexical) search in
  parallel against the same Qdrant collection, then fuses the ranked lists with
  Reciprocal Rank Fusion — rank-based, so no cross-scale score normalization is
  needed. BM25 IDF is computed server-side by Qdrant (`Modifier.IDF`).
- **Cross-encoder reranking.** The fused top-`candidate_k` (default 20) is re-scored
  by a cross-encoder that jointly encodes (query, passage); only the top-`top_k`
  reach the LLM. Disable with `RERANK_ENABLED=false` for a before/after comparison.
- **Citations done right.** Every answer references the chunks it used as `[n]`, and
  `/query` returns the corresponding `{id, source, chunk_index, score, snippet}`.
  The UI renders them as source cards under each answer.
- **Streaming.** `/query` with `stream: true` emits SSE `token` events, then a
  `citations` event, then `done`.

## Development

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on *nix
pip install -r requirements-dev.txt                # light: tests run against fakes
ruff check .
pytest
```

46 tests cover the chunker (overlap/coverage/determinism), RRF fusion properties,
reranker ordering, retrieval metrics, JWT auth (including 401 enforcement), the QA
set's shape, and `/ingest` → `/query` → `/eval` integration — all with fakes, so no
torch, no Qdrant, and no API key are needed.

## Deployment (free-tier path)

| Piece | Host |
|-------|------|
| FastAPI + embeddings + reranker | Hugging Face Spaces (Docker) |
| Vector store | Qdrant Cloud free tier (~1M vectors, hybrid built-in) |
| Chat UI | Served by the app at `/`, or deploy `frontend/` to Vercel with `localStorage.RAG_API` pointed at the Space |
| Eval gate | GitHub Actions (no hosting cost) |
| Generation | Gemini free tier |

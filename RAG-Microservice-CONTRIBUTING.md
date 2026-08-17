# Contributing to RAG Retrieval Microservice

Thanks for your interest in the project. Issues and pull requests are welcome — especially anything that improves retrieval quality, adds document format support, or strengthens the eval harness.

## Development setup

```bash
git clone https://github.com/Ritik650/RAG-Microservice.git
cd RAG-Microservice

python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on *nix
pip install -r requirements-dev.txt                # light — tests run against fakes, no torch/Qdrant needed
cp .env.example .env                                # add GEMINI_API_KEY if testing generation end-to-end
```

## Before opening a PR

```bash
ruff check .
pytest
```

The 46-test suite runs entirely against fakes (no torch, no live Qdrant, no API key required), so there's no excuse to skip it locally.

If your change touches retrieval (chunking, fusion, reranking, or embeddings), also run the evaluation gate against a live Qdrant instance:

```bash
docker compose up -d qdrant
pip install -r requirements.txt
python eval/run_ragas.py
```

**A retrieval regression is treated like a failing unit test.** If `hybrid_rerank` recall or MRR drops below the thresholds in `eval/run_ragas.py` (`EVAL_MIN_RECALL`, `EVAL_MIN_MRR`), CI fails the build. If your change intentionally trades retrieval quality for something else (latency, cost), say so explicitly in the PR description and update the thresholds in the same PR.

## Adding a new document format

Formats are registered in one place: [app/ingest/parsers.py](app/ingest/parsers.py). Adding a format is one parsing function plus one dict entry — it's automatically picked up by both `/ingest/files` and `scripts/load_folder.py`. Add a corresponding test in `tests/` covering at least one real sample of the format.

## Extending the QA eval set

[eval/qa_set.jsonl](eval/qa_set.jsonl) is hand-labeled — question → ground-truth → source triples over the sample corpus. If you add sample documents, add corresponding QA pairs so the harness keeps measuring real retrieval behavior rather than stale ground truth.

## Style

- Keep the parser registry pattern for new ingestion formats — don't special-case a format elsewhere in the pipeline.
- New endpoints should follow the existing pattern: Pydantic request/response models, and a corresponding entry in the API table in the README.
- Match the project's testing philosophy: use fakes for unit tests (fast, no external services), and reserve the real Qdrant/Gemini path for the eval gate.

## Security issues

Please don't open a public issue for security concerns — see [SECURITY.md](SECURITY.md) instead.

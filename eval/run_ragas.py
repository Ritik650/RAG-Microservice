"""Retrieval-quality regression gate (+ optional RAGAS generation eval).

Ingests sample_docs/ into Qdrant, runs the retrieval eval (dense vs hybrid vs
hybrid+rerank) on eval/qa_set.jsonl, prints a comparison table, and exits non-zero
if the hybrid_rerank configuration falls below the thresholds — this is what CI
runs to fail the build on retrieval regressions.

Usage:
    python eval/run_ragas.py [--corpus sample_docs] [--ragas] [--json out.json]

Env:
    QDRANT_URL (default http://localhost:6333)
    GEMINI_API_KEY (only needed with --ragas)

Thresholds (override via env):
    EVAL_MIN_RECALL   default 0.85   (recall@k, hybrid_rerank)
    EVAL_MIN_MRR      default 0.70   (MRR, hybrid_rerank)
    EVAL_MIN_FAITHFULNESS default 0.70 (RAGAS faithfulness, only with --ragas)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Allow running as `python eval/run_ragas.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.embed.embedder import Embedder  # noqa: E402
from app.embed.sparse import SparseEncoder  # noqa: E402
from app.evaluation.runner import load_qa_set, run_retrieval_eval  # noqa: E402
from app.ingest.pipeline import ingest_documents  # noqa: E402
from app.models import Document  # noqa: E402
from app.rerank.cross_encoder import CrossEncoderReranker  # noqa: E402
from app.store.qdrant_store import QdrantStore  # noqa: E402

MIN_RECALL = float(os.environ.get("EVAL_MIN_RECALL", "0.85"))
MIN_MRR = float(os.environ.get("EVAL_MIN_MRR", "0.70"))
MIN_FAITHFULNESS = float(os.environ.get("EVAL_MIN_FAITHFULNESS", "0.70"))


def load_corpus(folder: Path) -> list[Document]:
    docs = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                docs.append(Document(source=str(path.relative_to(folder)), text=text))
    return docs


def print_table(results: dict) -> None:
    k = results["k"]
    cols = [f"recall@{k}", "mrr", f"ndcg@{k}", "p50_ms", "p95_ms"]
    header = f"{'mode':<16}" + "".join(f"{c:>12}" for c in cols)
    print(f"\nRetrieval eval — {results['questions']} questions, k={k}")
    print(header)
    print("-" * len(header))
    for mode, metrics in results["modes"].items():
        print(f"{mode:<16}" + "".join(f"{metrics[c]:>12}" for c in cols))
    print()


async def run_ragas_eval(qa_items, embedder, sparse_encoder, store, reranker, settings) -> dict:
    """Generate answers with Gemini and score with RAGAS. Requires GEMINI_API_KEY
    and the optional eval dependencies (requirements-eval.txt)."""
    from datasets import Dataset
    from fastapi.concurrency import run_in_threadpool
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from app.generate.llm import GeminiGenerator
    from app.retrieve.hybrid import hybrid_search

    generator = GeminiGenerator(settings)
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in qa_items:
        candidates = await hybrid_search(
            item["question"], embedder, sparse_encoder, store,
            limit=settings.candidate_k, rrf_k=settings.rrf_k,
        )
        contexts = await run_in_threadpool(
            reranker.rerank, item["question"], candidates, settings.top_k
        )
        answer = await generator.generate(item["question"], contexts)
        rows["question"].append(item["question"])
        rows["answer"].append(answer)
        rows["contexts"].append([c["text"] for c in contexts])
        rows["ground_truth"].append(item["ground_truth"])

    result = evaluate(
        Dataset.from_dict(rows),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ChatGoogleGenerativeAI(model=settings.llm_model),
        embeddings=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
    )
    return {k: round(float(v), 4) for k, v in result.items()}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="sample_docs", help="folder of docs to ingest")
    parser.add_argument("--ragas", action="store_true", help="also run RAGAS generation eval")
    parser.add_argument("--json", default=None, help="write full results to this JSON file")
    args = parser.parse_args()

    settings = Settings()
    embedder = Embedder(settings.embed_model)
    sparse_encoder = SparseEncoder(settings.sparse_model)
    store = QdrantStore(settings)
    reranker = CrossEncoderReranker(settings.rerank_model)

    docs = load_corpus(Path(args.corpus))
    print(f"Ingesting {len(docs)} documents from {args.corpus}/ into {settings.qdrant_url} …")
    ingested = await ingest_documents(docs, embedder, sparse_encoder, store, settings)
    print(f"Upserted {ingested['chunks_upserted']} chunks.")

    qa_items = load_qa_set(settings.qa_set_path)
    results = await run_retrieval_eval(
        qa_items, embedder, sparse_encoder, store, reranker, settings
    )
    print_table(results)

    failures = []
    gated = results["modes"]["hybrid_rerank"]
    recall_key = f"recall@{results['k']}"
    if gated[recall_key] < MIN_RECALL:
        failures.append(f"{recall_key}={gated[recall_key]} < {MIN_RECALL}")
    if gated["mrr"] < MIN_MRR:
        failures.append(f"mrr={gated['mrr']} < {MIN_MRR}")

    if args.ragas:
        print("Running RAGAS generation eval (calls Gemini) …")
        ragas_scores = await run_ragas_eval(
            qa_items, embedder, sparse_encoder, store, reranker, settings
        )
        results["ragas"] = ragas_scores
        print("RAGAS:", json.dumps(ragas_scores, indent=2))
        if ragas_scores.get("faithfulness", 1.0) < MIN_FAITHFULNESS:
            failures.append(
                f"faithfulness={ragas_scores['faithfulness']} < {MIN_FAITHFULNESS}"
            )

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")

    await store.close()

    if failures:
        print("EVAL GATE FAILED:", "; ".join(failures), file=sys.stderr)
        return 1
    print(f"EVAL GATE PASSED ({recall_key} >= {MIN_RECALL}, mrr >= {MIN_MRR}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

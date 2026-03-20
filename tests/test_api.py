"""Integration tests for the API using fakes for the embedders, store, reranker, and LLM.

No network, no torch, no Gemini key required.
"""

from fastapi.testclient import TestClient

from app.main import (
    app,
    get_embedder,
    get_generator,
    get_reranker,
    get_sparse_encoder,
    get_store,
)


class FakeEmbedder:
    dim = 3

    def encode(self, texts):
        # Deterministic, dimensionality-correct vectors; values are irrelevant to the fake store.
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeSparseEncoder:
    def encode_docs(self, texts):
        return [([1, 2], [0.5, 0.5]) for _ in texts]

    def encode_query(self, text):
        return [1, 2], [1.0, 1.0]


class FakeStore:
    def __init__(self):
        self.points = {}

    async def ensure_ready(self, dim):
        pass

    async def upsert(self, points):
        for p in points:
            self.points[p["id"]] = p

    def _results(self):
        return [
            {
                "id": "france.md::0",
                "source": "france.md",
                "chunk_index": 0,
                "text": "Paris is the capital of France.",
                "score": 0.91,
            }
        ]

    async def search_dense(self, vector, top_k):
        return self._results()[:top_k]

    async def search_sparse(self, indices, values, top_k):
        return self._results()[:top_k]

    async def count(self):
        return len(self.points)


class FakeReranker:
    def rerank(self, question, hits, top_k):
        return hits[:top_k]


class FakeGenerator:
    async def generate(self, question, contexts):
        return "Paris is the capital of France. [1]"

    async def stream(self, question, contexts):
        for token in ["Paris ", "is the ", "capital. ", "[1]"]:
            yield token


def make_client():
    store = FakeStore()
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[get_sparse_encoder] = lambda: FakeSparseEncoder()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_generator] = lambda: FakeGenerator()
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    return TestClient(app), store


def teardown_function():
    app.dependency_overrides.clear()


def test_healthz():
    client, _ = make_client()
    with client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ingest_then_query_json():
    client, store = make_client()
    with client:
        doc = {"source": "france.md", "text": "Paris is the capital of France."}
        ingest = client.post("/ingest", json={"documents": [doc]})
        assert ingest.status_code == 200
        assert ingest.json()["chunks_upserted"] == 1
        # Sparse BM25 weights are stored alongside the dense vector.
        (point,) = store.points.values()
        assert point["sparse_indices"] == [1, 2]

        query = client.post("/query", json={"question": "What is the capital of France?"})
    assert query.status_code == 200
    body = query.json()
    assert "Paris" in body["answer"]
    assert body["citations"][0]["source"] == "france.md"
    assert body["citations"][0]["snippet"].startswith("Paris")


def test_ingest_is_idempotent():
    client, store = make_client()
    doc = {"documents": [{"source": "france.md", "text": "Paris is the capital of France."}]}
    with client:
        client.post("/ingest", json=doc)
        first = set(store.points)
        client.post("/ingest", json=doc)
        second = set(store.points)
    # Re-ingesting the same document produces identical point ids -> no duplicates.
    assert first == second
    assert len(second) == 1


def test_ingest_files_upload():
    client, store = make_client()
    with client:
        resp = client.post(
            "/ingest/files",
            files=[
                ("files", ("notes.txt", b"Paris is the capital of France.", "text/plain")),
                ("files", ("page.html", b"<p>Berlin is the capital of Germany.</p>", "text/html")),
            ],
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["documents"] == 2
    assert body["chunks_upserted"] == 2
    sources = {p["source"] for p in store.points.values()}
    assert sources == {"notes.txt", "page.html"}


def test_ingest_files_rejects_unsupported_type():
    client, _ = make_client()
    with client:
        resp = client.post(
            "/ingest/files",
            files=[("files", ("archive.zip", b"PK\x03\x04", "application/zip"))],
        )
    assert resp.status_code == 415
    assert "Unsupported file type" in resp.json()["detail"]


def test_query_deduplicates_hybrid_results():
    # The fake store returns the same hit from both dense and sparse search;
    # RRF fusion must collapse it to a single citation, not two.
    client, _ = make_client()
    with client:
        resp = client.post("/query", json={"question": "capital of France?"})
    assert resp.status_code == 200
    citations = resp.json()["citations"]
    assert len(citations) == 1
    assert citations[0]["id"] == "france.md::0"


def test_query_streaming_sse():
    client, _ = make_client()
    with client:
        resp = client.post("/query", json={"question": "capital of France?", "stream": True})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "event: token" in body
    assert "Paris" in body
    assert "event: citations" in body
    assert "event: done" in body

"""Integration tests for the API using fakes for the embedder, store, and LLM.

No network, no torch, no Gemini key required.
"""

from fastapi.testclient import TestClient

from app.main import app, get_embedder, get_generator, get_store


class FakeEmbedder:
    dim = 3

    def encode(self, texts):
        # Deterministic, dimensionality-correct vectors; values are irrelevant to the fake store.
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeStore:
    def __init__(self):
        self.points = {}

    async def ensure_ready(self, dim):
        pass

    async def upsert(self, points):
        for p in points:
            self.points[p["id"]] = p

    async def search(self, vector, top_k):
        results = [
            {
                "id": "france.md::0",
                "source": "france.md",
                "chunk_index": 0,
                "text": "Paris is the capital of France.",
                "score": 0.91,
            }
        ]
        return results[:top_k]

    async def count(self):
        return len(self.points)


class FakeGenerator:
    async def generate(self, question, contexts):
        return "Paris is the capital of France. [1]"

    async def stream(self, question, contexts):
        for token in ["Paris ", "is the ", "capital. ", "[1]"]:
            yield token


def make_client():
    store = FakeStore()
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_generator] = lambda: FakeGenerator()
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
        ingest = client.post(
            "/ingest",
            json={"documents": [{"source": "france.md", "text": "Paris is the capital of France."}]},
        )
        assert ingest.status_code == 200
        assert ingest.json()["chunks_upserted"] == 1

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
        first = store.count.__self__.points.copy()
        client.post("/ingest", json=doc)
        second = store.count.__self__.points
    # Re-ingesting the same document produces identical point ids -> no duplicates.
    assert set(first) == set(second)
    assert len(second) == 1


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

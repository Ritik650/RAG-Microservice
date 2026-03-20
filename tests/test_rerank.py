from app.rerank.cross_encoder import CrossEncoderReranker


class FakeModel:
    """Scores each passage by how many query words it contains."""

    def predict(self, pairs):
        return [sum(w in text for w in question.split()) for question, text in pairs]


def make_reranker():
    r = CrossEncoderReranker("fake-model")
    r._model = FakeModel()  # bypass lazy load; no torch in tests
    return r


def test_reranker_orders_by_relevance_and_cuts_to_top_k():
    hits = [
        {"id": "a", "text": "nothing relevant here", "score": 0.9},
        {"id": "b", "text": "capital of France", "score": 0.1},
        {"id": "c", "text": "France", "score": 0.5},
    ]
    out = make_reranker().rerank("capital of France", hits, top_k=2)
    assert [h["id"] for h in out] == ["b", "c"]
    # Scores are replaced with the cross-encoder's relevance scores.
    assert out[0]["score"] >= out[1]["score"]


def test_reranker_handles_empty_input():
    assert make_reranker().rerank("anything", [], top_k=5) == []

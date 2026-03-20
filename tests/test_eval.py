from tests.test_api import make_client


def test_eval_endpoint_returns_all_modes():
    client, _ = make_client()
    with client:
        resp = client.post("/eval")
    assert resp.status_code == 200
    body = resp.json()
    assert body["questions"] > 0
    assert set(body["modes"]) == {"dense", "hybrid", "hybrid_rerank"}
    k = body["k"]
    for metrics in body["modes"].values():
        for key in (f"recall@{k}", "mrr", f"ndcg@{k}", "p50_ms", "p95_ms"):
            assert key in metrics


def test_qa_set_is_well_formed():
    from app.evaluation.runner import load_qa_set

    items = load_qa_set("eval/qa_set.jsonl")
    assert len(items) >= 20, "spec asks for a meaningful hand-labeled set"
    for item in items:
        assert item["question"].strip()
        assert item["ground_truth"].strip()
        assert isinstance(item["sources"], list) and item["sources"]

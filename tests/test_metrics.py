import pytest

from app.evaluation.metrics import aggregate, ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a"}, 3) == 1.0
    assert recall_at_k(["x", "a"], {"a"}, 1) == 0.0
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, 3) == 1.0
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5
    assert recall_at_k([], {"a"}, 5) == 0.0
    assert recall_at_k(["a"], set(), 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)


def test_ndcg_penalizes_late_relevance():
    early = ndcg_at_k(["a", "x", "y"], {"a"}, 3)
    late = ndcg_at_k(["x", "y", "a"], {"a"}, 3)
    assert early == pytest.approx(1.0)
    assert 0 < late < early


def test_ndcg_no_relevant_is_zero():
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_ndcg_never_exceeds_one_with_duplicate_sources():
    # Many chunks from the same relevant source must not inflate the score:
    # each expected source is credited once, at its first retrieved rank.
    assert ndcg_at_k(["a", "a", "a", "a", "a"], {"a"}, 5) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "a", "a"], {"a"}, 3) == pytest.approx(1.0 / __import__("math").log2(3))


def test_aggregate_averages_per_query_metrics():
    agg = aggregate([{"mrr": 1.0}, {"mrr": 0.5}])
    assert agg == {"mrr": 0.75}
    assert aggregate([]) == {}

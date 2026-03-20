import pytest

from app.retrieve.fusion import rrf


def test_doc_in_both_lists_outranks_single_list_docs():
    fused = rrf([["a", "b", "c"], ["b", "d", "e"]])
    assert fused[0][0] == "b"


def test_rank_order_within_a_single_list_is_preserved():
    fused = rrf([["a", "b", "c"]])
    assert [doc_id for doc_id, _ in fused] == ["a", "b", "c"]


def test_scores_follow_the_rrf_formula():
    k = 60
    fused = dict(rrf([["a"], ["a"]], k=k))
    assert fused["a"] == pytest.approx(2 / (k + 1))


def test_deterministic_tie_break_by_id():
    # "x" and "y" get identical scores; ties break lexicographically.
    fused = rrf([["y"], ["x"]])
    assert [doc_id for doc_id, _ in fused] == ["x", "y"]


def test_empty_input():
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_negative_k_rejected():
    with pytest.raises(ValueError):
        rrf([["a"]], k=-1)


def test_fusion_is_deterministic():
    lists = [["a", "b", "c"], ["c", "a", "d"]]
    assert rrf(lists) == rrf(lists)

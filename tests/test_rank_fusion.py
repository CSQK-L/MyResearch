"""Pure fusion tests: run with --noconftest -c /dev/null, without Django or models."""

from copy import deepcopy

import pytest

from khoj.search_type.retrieval_types import RetrievalHit


def fuse(dense, bm25, **kwargs):
    from khoj.search_type.rank_fusion import reciprocal_rank_fusion

    return reciprocal_rank_fusion(dense, bm25, **kwargs)


def hits(*ids):
    return [RetrievalHit(entry_id=i) for i in ids]


@pytest.mark.parametrize(
    "dense,bm25,expected",
    [
        ([2, 1], [], [2, 1]),
        ([], [2, 1], [2, 1]),
        ([1, 2], [2, 3], [2, 1, 3]),
        ([2], [1], [1, 2]),
        ([2, 1], [2, 1], [2, 1]),
        ([], [], []),
    ],
)
def test_membership_and_order(dense, bm25, expected):
    assert [h.entry_id for h in fuse(hits(*dense), hits(*bm25))] == expected


def test_formula_ranks_scores_and_no_mutation():
    dense = [RetrievalHit(1, dense_rank=90, cosine_distance=0.2), RetrievalHit(2, cosine_distance=0.4)]
    lexical = [RetrievalHit(3, bm25_score=7), RetrievalHit(2, bm25_score=4), RetrievalHit(1, bm25_score=2)]
    before = deepcopy((dense, lexical))
    result = fuse(dense, lexical)
    a = next(h for h in result if h.entry_id == 1)
    assert a.rrf_score == pytest.approx(1 / 61 + 1 / 63)
    assert (a.dense_rank, a.bm25_rank, a.cosine_distance, a.bm25_score) == (1, 3, 0.2, 2)
    assert (dense, lexical) == before
    assert all(h is not original for h in result for original in dense + lexical)


def test_weights_top_k_and_duplicate_best_rank():
    assert [h.entry_id for h in fuse(hits(1), hits(2), bm25_weight=2, top_k=1)] == [2]
    result = fuse(hits(1, 1, 2), hits(2, 2))
    assert result[0].entry_id == 2
    assert result[0].rrf_score == pytest.approx(1 / 63 + 1 / 61)
    assert result[1].rrf_score == pytest.approx(1 / 61)
    assert fuse(hits(1), [], top_k=0) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"k": -1},
        {"k": True},
        {"k": float("nan")},
        {"dense_weight": -1},
        {"bm25_weight": float("inf")},
        {"dense_weight": "1"},
        {"top_k": -1},
        {"top_k": True},
        {"top_k": 1.5},
        {"k": None},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        fuse([], [], **kwargs)

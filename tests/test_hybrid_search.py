"""Orchestration failures and independent recall; only backend calls are mocked."""

from unittest.mock import AsyncMock

import pytest
from asgiref.sync import async_to_sync
from django.db import OperationalError

from khoj.search_type.retrieval_types import RetrievalHit


def test_independent_recall_and_overlap(monkeypatch):
    from khoj.search_type import hybrid_search

    dense = AsyncMock(
        return_value=[type("Row", (), {"pk": 1, "distance": 0.1})(), type("Row", (), {"pk": 2, "distance": 0.2})()]
    )
    monkeypatch.setattr(hybrid_search.text_search, "query", dense)
    calls = []

    def lexical(**kwargs):
        calls.append(kwargs)
        return [RetrievalHit(2, bm25_score=9), RetrievalHit(3, bm25_score=4)]

    monkeypatch.setattr(hybrid_search.EntryAdapters, "search_with_bm25", lexical)
    result = async_to_sync(hybrid_search.query)('alpha file:"*.org"', None)
    assert [h.entry_id for h in result] == [2, 1, 3]
    assert dense.call_args.kwargs["top_k"] == 40
    assert calls[0]["max_results"] == 40
    assert calls[0]["raw_query"] == 'alpha file:"*.org"'


@pytest.mark.parametrize(
    "dense_state,bm25_state,expected",
    [("ok", "empty", [1]), ("empty", "ok", [2]), ("empty", "empty", []), ("error", "ok", [2]), ("ok", "error", [1])],
)
def test_empty_and_degraded_branches(monkeypatch, caplog, dense_state, bm25_state, expected):
    from khoj.search_type import hybrid_search

    dense = AsyncMock(return_value=[] if dense_state == "empty" else [type("Row", (), {"pk": 1, "distance": 0.1})()])
    if dense_state == "error":
        dense.side_effect = OperationalError("dense unavailable")
    monkeypatch.setattr(hybrid_search.text_search, "query", dense)

    def lexical(**kwargs):
        if bm25_state == "error":
            raise OperationalError("missing pg_search/index or query timeout")
        return [] if bm25_state == "empty" else [RetrievalHit(2, bm25_score=3)]

    monkeypatch.setattr(hybrid_search.EntryAdapters, "search_with_bm25", lexical)
    assert [h.entry_id for h in async_to_sync(hybrid_search.query)("alpha", None)] == expected
    assert bool(caplog.records) == ("error" in (dense_state, bm25_state))


def test_both_fail_and_permission_errors_propagate(monkeypatch):
    from khoj.search_type import hybrid_search

    monkeypatch.setattr(hybrid_search.text_search, "query", AsyncMock(side_effect=OperationalError("dense")))

    def broken(**kwargs):
        raise OperationalError("lexical")

    monkeypatch.setattr(hybrid_search.EntryAdapters, "search_with_bm25", broken)
    with pytest.raises(hybrid_search.RetrievalUnavailable):
        async_to_sync(hybrid_search.query)("alpha", None)

    def denied(**kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(hybrid_search.EntryAdapters, "search_with_bm25", denied)
    with pytest.raises(PermissionError):
        async_to_sync(hybrid_search.query)("alpha", None)

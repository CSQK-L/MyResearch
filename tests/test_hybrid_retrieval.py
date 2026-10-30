"""Full execute_search path with real pgvector and controlled lexical recall."""

import asyncio

import pytest
from asgiref.sync import async_to_sync
from django.db import OperationalError

from khoj.database.adapters import EntryAdapters
from khoj.routers.helpers import execute_search
from khoj.search_type.retrieval_types import RetrievalHit
from khoj.utils import state
from tests import test_retrieval as client_fixtures
from tests import test_retrieval as retrieval_fixtures
from tests.test_retrieval import make_entry

retrieval_client = client_fixtures.retrieval_client
retrieval_env = retrieval_fixtures.retrieval_env

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture(autouse=True)
def vector_default(monkeypatch):
    monkeypatch.delenv("KHOJ_RETRIEVAL_MODE", raising=False)


@pytest.fixture
def lexical_only(retrieval_env, monkeypatch):
    dense = make_entry(user=retrieval_env, raw="semantic alpha", compiled="semantic alpha")
    lexical = make_entry(
        user=retrieval_env, raw="lexical evidence", compiled="lexical evidence", embeddings=[0.0, 1.0, 0.0]
    )
    monkeypatch.setattr(
        EntryAdapters, "search_with_bm25", lambda **kwargs: [RetrievalHit(lexical.pk, bm25_rank=1, bm25_score=7)]
    )
    return dense, lexical


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("vector", ["semantic alpha"]),
        ("bm25", ["lexical evidence"]),
        ("hybrid", ["semantic alpha", "lexical evidence"]),
    ],
)
def test_modes_and_real_distance(retrieval_env, lexical_only, mode, expected):
    rows = async_to_sync(execute_search)(retrieval_env, "alpha", max_distance=0.1, mode=mode)
    assert [row.entry for row in rows] == expected
    assert [row.score for row in rows] == ([0.0, 1.0] if mode == "hybrid" else [1.0] if mode == "bm25" else [0.0])
    assert all(isinstance(row.model_dump(mode="json")["score"], float) for row in rows)


def test_rerank_after_fusion_ties_and_final_limit(retrieval_env, lexical_only, monkeypatch):
    name = next(iter(state.cross_encoder_model))
    seen = []

    def predict(query, rows):
        seen.append([row.entry for row in rows])
        return [0.1 if row.entry == "semantic alpha" else 0.9 for row in rows]

    monkeypatch.setattr(state.cross_encoder_model[name], "predict", predict)
    result = async_to_sync(execute_search)(retrieval_env, "alpha", n=1, r=True, max_distance=0.1, mode="hybrid")
    assert seen == [["semantic alpha", "lexical evidence"]]
    assert result[0].entry == "lexical evidence"
    monkeypatch.setattr(state.cross_encoder_model[name], "predict", lambda q, rows: [0.5] * len(rows))
    result = async_to_sync(execute_search)(retrieval_env, "alpha", r=True, max_distance=0.1, mode="hybrid")
    assert [row.entry for row in result] == ["semantic alpha", "lexical evidence"]


def test_chunks_are_reranked_before_legacy_dedupe(retrieval_env, lexical_only, monkeypatch):
    dense, lexical = lexical_only
    from khoj.database.models import Entry

    Entry.objects.filter(pk=lexical.pk).update(corpus_id=dense.corpus_id)
    model = next(iter(state.cross_encoder_model.values()))
    monkeypatch.setattr(model, "predict", lambda q, rows: [0.1, 0.9])
    rows = async_to_sync(execute_search)(retrieval_env, "alpha", r=True, max_distance=0.1, mode="hybrid")
    assert [row.entry for row in rows] == ["lexical evidence"]


def test_no_result_cache_across_modes_or_updates(retrieval_env, lexical_only):
    from khoj.database.models import Entry

    rows = async_to_sync(execute_search)(retrieval_env, "alpha", max_distance=0.1, mode="vector")
    assert [r.entry for r in rows] == ["semantic alpha"]
    rows = async_to_sync(execute_search)(retrieval_env, "alpha", max_distance=0.1, mode="hybrid")
    assert len(rows) == 2
    Entry.objects.filter(user=retrieval_env).delete()
    assert async_to_sync(execute_search)(retrieval_env, "alpha", mode="vector") == []
    assert async_to_sync(execute_search)(retrieval_env, "alpha", mode="hybrid") == []


def test_environment_mode_and_explicit_override(retrieval_env, lexical_only, monkeypatch):
    monkeypatch.setenv("KHOJ_RETRIEVAL_MODE", "hybrid")
    assert len(async_to_sync(execute_search)(retrieval_env, "alpha", max_distance=0.1)) == 2
    assert len(async_to_sync(execute_search)(retrieval_env, "alpha", max_distance=0.1, mode="vector")) == 1
    monkeypatch.setenv("KHOJ_RETRIEVAL_MODE", "bad")
    with pytest.raises(ValueError, match="KHOJ_RETRIEVAL_MODE"):
        async_to_sync(execute_search)(retrieval_env, "alpha")


def test_dense_failure_keeps_bm25_if_distance_can_be_hydrated(retrieval_env, lexical_only, monkeypatch):
    def broken(*args, **kwargs):
        raise OperationalError("dense database failure")

    monkeypatch.setattr(EntryAdapters, "search_with_embeddings", broken)
    rows = async_to_sync(execute_search)(retrieval_env, "alpha", mode="hybrid")
    assert [(r.entry, r.score) for r in rows] == [("lexical evidence", 1.0)]


def test_inference_off_event_loop(retrieval_env, lexical_only, monkeypatch):
    name = next(iter(state.embeddings_model))
    worker_threads = []

    def embed(query):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        worker_threads.append(True)
        assert query == "alpha"
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(state.embeddings_model[name], "embed_query", embed)
    async_to_sync(execute_search)(retrieval_env, 'dt>="2024-01-01" alpha -file:"secret*"', mode="bm25")
    assert worker_threads


def test_chat_dedupe_preserves_scores_and_query_provenance(retrieval_env, lexical_only):
    from khoj.search_type import text_search

    rows = async_to_sync(execute_search)(retrieval_env, "alpha", mode="hybrid", max_distance=0.1, r=True)
    first = next(row for row in rows if row.entry == "lexical evidence")
    first.additional["query"] = "alpha"
    second = first.model_copy(deep=True)
    second.additional["query"] = "second"
    result = list(text_search.deduplicated_search_responses([first, second]))
    assert len(result) == 1
    assert result[0].additional["entry_id"] == lexical_only[1].pk
    assert result[0].cross_score == first.cross_score
    assert result[0].additional["queries"] == ["alpha", "second"]
    assert "queries" not in first.additional


def test_invalid_candidate_distance_and_missing_embedding_model(retrieval_env, lexical_only, monkeypatch):
    from khoj.database.models import Entry
    from khoj.search_type.hybrid_search import RetrievalUnavailable

    Entry.objects.filter(pk=lexical_only[1].pk).update(embeddings=[0.0, 0.0, 0.0])
    with pytest.raises(RetrievalUnavailable, match="cosine distance"):
        async_to_sync(execute_search)(retrieval_env, "alpha", mode="bm25")
    monkeypatch.setattr(state, "embeddings_model", {})
    with pytest.raises(RetrievalUnavailable, match="Query embedding unavailable"):
        async_to_sync(execute_search)(retrieval_env, "alpha", mode="hybrid")

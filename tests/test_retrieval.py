"""PR-1 retrieval contracts, using real pgvector queries without model downloads."""

import json
import math
import uuid
from types import SimpleNamespace

import pytest
from asgiref.sync import async_to_sync
from fastapi import FastAPI
from fastapi.testclient import TestClient

from khoj.database.adapters import AgentAdapters, EntryAdapters, get_default_search_model
from khoj.database.models import Agent, Entry, EntryDates
from khoj.routers import api
from khoj.routers.helpers import execute_search
from khoj.search_type import text_search
from khoj.utils import state
from khoj.utils.helpers import LRU
from khoj.utils.state import SearchType
from tests.helpers import ChatModelFactory, UserFactory

pytestmark = pytest.mark.django_db(transaction=True)


def test_retrieval_types():
    from khoj.search_type.retrieval_types import RetrievalHit, RetrievalMode, RetrievalOptions

    assert [m.value for m in RetrievalMode] == ["vector", "bm25", "hybrid"]
    assert RetrievalMode.VECTOR == "vector"
    assert json.dumps({"mode": RetrievalMode.VECTOR}) == '{"mode": "vector"}'
    with pytest.raises(ValueError):
        RetrievalMode("invalid")
    assert RetrievalMode(str("vector")) is RetrievalMode.VECTOR
    assert RetrievalOptions().mode is RetrievalMode.VECTOR
    assert RetrievalOptions().dense_top_k == 10
    assert RetrievalOptions(mode="bm25").mode is RetrievalMode.BM25
    with pytest.raises(ValueError):
        RetrievalOptions(mode="invalid")
    hit = RetrievalHit(entry_id=123, dense_rank=1, cosine_distance=0.25)
    assert hit.entry_id == 123
    assert hit.bm25_rank is hit.bm25_score is None
    assert hit.rrf_score == 0.0
    empty = RetrievalHit(entry_id=124)
    assert empty.dense_rank is empty.cosine_distance is empty.bm25_rank is empty.bm25_score is None


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True, "30", None])
def test_invalid_retrieval_options(top_k):
    from khoj.search_type.retrieval_types import RetrievalOptions

    with pytest.raises(ValueError, match="positive integer"):
        RetrievalOptions(dense_top_k=top_k)


@pytest.fixture
def retrieval_env(monkeypatch, default_user):
    search_model = get_default_search_model()
    search_model.bi_encoder_confidence_threshold = 0.5
    search_model.save()
    # Only model inference is replaced; DB, filtering, collation, reranking and routes are real.
    monkeypatch.setattr(
        state, "embeddings_model", {search_model.name: SimpleNamespace(embed_query=lambda q: [1.0, 0.0, 0.0])}
    )
    monkeypatch.setattr(
        state,
        "cross_encoder_model",
        {
            search_model.name: SimpleNamespace(
                inference_server_enabled=lambda: False,
                predict=lambda q, hits: [i / len(hits) for i in range(len(hits))],
            )
        },
    )
    monkeypatch.setattr(state, "SearchType", SearchType)
    monkeypatch.setitem(state.query_cache, default_user.uuid, LRU())
    return default_user


def make_entry(user=None, agent=None, index=0, **kwargs):
    values = dict(
        user=user,
        agent=agent,
        embeddings=[1.0, 0.0, 0.0],
        raw=f"alpha content {index}",
        compiled=f"compiled {index}",
        hashed_value=str(uuid.uuid4()),
        file_type="org",
        file_path=f"notes/{index}.org",
    )
    values.update(kwargs)
    return Entry.objects.bulk_create([Entry(**values)])[0]


@pytest.mark.parametrize("top_k", [1, 5, 10, 30, 100])
def test_dynamic_top_k(retrieval_env, top_k):
    for i in range(35):
        make_entry(user=retrieval_env, index=i)
    hits = async_to_sync(text_search.query)("alpha", retrieval_env, top_k=top_k)
    assert len(hits) == min(35, top_k)
    assert len({h.pk for h in hits}) == len(hits)
    assert all(isinstance(h, Entry) for h in hits)


def test_query_default_and_old_positional_arguments(retrieval_env):
    for i in range(12):
        make_entry(user=retrieval_env, index=i)
    assert len(async_to_sync(text_search.query)("alpha", retrieval_env)) == 10
    assert (
        len(async_to_sync(text_search.query)("alpha", retrieval_env, SearchType.All, [1.0, 0.0, 0.0], math.inf, None))
        == 10
    )


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_query_rejects_invalid_top_k(retrieval_env, top_k):
    with pytest.raises(ValueError, match="positive integer"):
        async_to_sync(text_search.query)("alpha", retrieval_env, top_k=top_k)


def test_hit_conversion_preserves_chunks_and_old_collation(retrieval_env):
    corpus_id = uuid.uuid4()
    entries = [make_entry(user=retrieval_env, index=i, corpus_id=corpus_id) for i in range(2)]
    rows = list(EntryAdapters.search_with_embeddings("alpha", [1.0, 0.0, 0.0], retrieval_env))
    hits = text_search.to_retrieval_hits(rows)
    assert {h.entry_id for h in hits} == {e.pk for e in entries}
    assert all(isinstance(h.entry_id, int) for h in hits)
    assert [h.dense_rank for h in hits] == [1, 2]
    assert all(h.cosine_distance == 0.0 and h.bm25_score is None for h in hits)
    assert text_search.to_retrieval_hits([]) == []
    assert len(list(text_search.collate_results(rows))) == 1
    assert len(list(text_search.collate_results(rows, dedupe=False))) == 2
    assert list(text_search.collate_results(rows))[0].score == 0.0


@pytest.mark.parametrize(
    "query",
    [
        "alpha",
        '+"alpha"',
        '-"beta"',
        'file:"notes/*.org"',
        'dt>="2024-01-01" dt<="2024-12-31"',
        '+"alpha" -"beta" file:"notes/*.org" dt>="2024-01-01"',
    ],
)
def test_scopes_and_filters(retrieval_env, query):
    other = UserFactory()
    agent = Agent.objects.create(name="Scope", creator=retrieval_env, chat_model=ChatModelFactory())
    unrelated = Agent.objects.create(name="Other", creator=other, chat_model=ChatModelFactory())
    own = make_entry(user=retrieval_env)
    agent_entry = make_entry(agent=agent)
    foreign = make_entry(user=other)
    unrelated_entry = make_entry(agent=unrelated)
    for e in [own, agent_entry, foreign, unrelated_entry]:
        EntryDates.objects.create(entry=e, date="2024-06-01")
    for user, scope, expected in [
        (retrieval_env, None, {own.pk}),
        (None, agent, {agent_entry.pk}),
        (retrieval_env, agent, {own.pk, agent_entry.pk}),
        (None, None, set()),
    ]:
        rows = EntryAdapters.search_with_embeddings(query, [1.0, 0.0, 0.0], user, agent=scope)
        assert {e.pk for e in rows} == expected


@pytest.mark.parametrize("query", ["alpha", '+"alpha"', 'file:"notes/*"'])
def test_file_type_filter_always_applies(retrieval_env, query):
    org = make_entry(user=retrieval_env)
    make_entry(user=retrieval_env, file_type="pdf", file_path="notes/a.pdf")
    assert list(EntryAdapters.apply_filters(retrieval_env, query, "org").values_list("pk", flat=True)) == [org.pk]


def test_combined_filter_excludes_nonmatching_entries(retrieval_env):
    wanted = make_entry(user=retrieval_env)
    wrong_word = make_entry(user=retrieval_env, raw="alpha beta")
    wrong_path = make_entry(user=retrieval_env, file_path="private/0.org")
    wrong_date = make_entry(user=retrieval_env)
    for e in [wanted, wrong_word, wrong_path]:
        EntryDates.objects.create(entry=e, date="2024-06-01")
    EntryDates.objects.create(entry=wrong_date, date="2020-01-01")
    query = '+"alpha" -"beta" file:"notes/*" dt>="2024-01-01" dt<="2024-12-31"'
    assert list(EntryAdapters.apply_filters(retrieval_env, query).values_list("pk", flat=True)) == [wanted.pk]


def test_date_join_does_not_duplicate_candidates(retrieval_env):
    first = make_entry(user=retrieval_env)
    second = make_entry(user=retrieval_env, index=1)
    for e in [first, second]:
        for day in ["2024-06-01", "2024-07-01"]:
            EntryDates.objects.create(entry=e, date=day)
    rows = list(EntryAdapters.search_with_embeddings('dt>="2024-01-01"', [1.0, 0.0, 0.0], retrieval_env, max_results=2))
    assert len({e.pk for e in rows}) == 2


@pytest.mark.parametrize("threshold, expected", [(None, 1), (0.0, 1), (math.inf, 2)])
def test_query_distance_threshold(retrieval_env, threshold, expected):
    make_entry(user=retrieval_env)
    make_entry(user=retrieval_env, embeddings=[0.0, 1.0, 0.0])
    assert len(async_to_sync(text_search.query)("alpha", retrieval_env, max_distance=threshold)) == expected


def test_zero_threshold_is_not_replaced(retrieval_env):
    make_entry(user=retrieval_env, embeddings=[1.0, 1.0, 0.0])
    assert async_to_sync(text_search.query)("alpha", retrieval_env, max_distance=0.0) == []


def test_model_zero_threshold_is_preserved(retrieval_env):
    model = get_default_search_model()
    model.bi_encoder_confidence_threshold = 0.0
    model.save()
    make_entry(user=retrieval_env, embeddings=[1.0, 1.0, 0.0])
    assert async_to_sync(text_search.query)("alpha", retrieval_env) == []


@pytest.fixture
def retrieval_client(retrieval_env, monkeypatch):
    from starlette.authentication import AuthCredentials, AuthenticationBackend, SimpleUser
    from starlette.middleware.authentication import AuthenticationMiddleware

    class TestAuth(AuthenticationBackend):
        async def authenticate(self, conn):
            user = SimpleUser("retrieval-test")
            user.object = retrieval_env
            return AuthCredentials(["authenticated"]), user

    monkeypatch.setattr(api, "update_telemetry_state", lambda **kwargs: None)
    app = FastAPI()
    app.include_router(api.api, prefix="/api")
    app.add_middleware(AuthenticationMiddleware, backend=TestAuth())
    return TestClient(app)


@pytest.mark.parametrize("n, expected", [(None, 5), (5, 5), (30, 30), (100, 35)])
@pytest.mark.parametrize("rerank", [False, True])
def test_api_dynamic_count_and_score(retrieval_client, retrieval_env, n, expected, rerank):
    for i in range(35):
        make_entry(user=retrieval_env, index=i)
    params = {"q": "alpha", "r": str(rerank).lower()}
    if n is not None:
        params["n"] = n
    response = retrieval_client.get("/api/search", params=params)
    assert response.status_code == 200, response.text
    rows = response.json()
    assert len(rows) == expected
    assert all(float(row["score"]) == 0.0 for row in rows)
    if rerank:
        assert [row["cross-score"] for row in rows] == sorted(row["cross-score"] for row in rows)


@pytest.mark.parametrize("n", [0, -1, 1001, "1.5", "bad"])
def test_api_invalid_count(retrieval_client, n):
    response = retrieval_client.get("/api/search", params={"q": "alpha", "n": n})
    assert response.status_code == 422


def test_api_zero_distance(retrieval_client, retrieval_env):
    make_entry(user=retrieval_env, embeddings=[1.0, 1.0, 0.0])
    response = retrieval_client.get("/api/search", params={"q": "alpha", "max_distance": 0.0})
    assert response.status_code == 200
    assert response.json() == []


def test_private_agent_access_and_cache_scope(retrieval_env):
    other = UserFactory()
    private = Agent.objects.create(
        name="Private", creator=other, chat_model=ChatModelFactory(), privacy_level="private"
    )
    allowed = Agent.objects.create(
        name="Allowed", creator=retrieval_env, chat_model=ChatModelFactory(), privacy_level="private"
    )
    second_allowed = Agent.objects.create(
        name="Second allowed", creator=retrieval_env, chat_model=ChatModelFactory(), privacy_level="private"
    )
    second_entry = make_entry(agent=second_allowed)
    agent_entry = make_entry(agent=allowed)
    make_entry(agent=private)
    assert not async_to_sync(AgentAdapters.ais_agent_accessible)(private, retrieval_env)
    assert async_to_sync(execute_search)(retrieval_env, "alpha", agent=private) == []
    assert async_to_sync(execute_search)(retrieval_env, "alpha") == []
    rows = async_to_sync(execute_search)(retrieval_env, "alpha", agent=allowed)
    assert len(rows) == 1
    assert rows[0].corpus_id == str(agent_entry.corpus_id)
    second_rows = async_to_sync(execute_search)(retrieval_env, "alpha", agent=second_allowed)
    assert len(second_rows) == 1
    assert second_rows[0].corpus_id == str(second_entry.corpus_id)
    assert async_to_sync(execute_search)(retrieval_env, "alpha") == []


@pytest.mark.parametrize("n", [0, -1, 1001, 1.5, True, "30"])
def test_execute_search_rejects_invalid_count_before_encoding(retrieval_env, monkeypatch, n):
    def unexpected_encode(query):
        pytest.fail("invalid n reached model inference")

    class UnexpectedCache:
        def __contains__(self, key):
            pytest.fail("invalid n reached cache access")

    model = get_default_search_model()
    monkeypatch.setitem(state.query_cache, retrieval_env.uuid, UnexpectedCache())
    monkeypatch.setattr(state.embeddings_model[model.name], "embed_query", unexpected_encode)
    with pytest.raises(ValueError, match="between 1 and 1000"):
        async_to_sync(execute_search)(retrieval_env, "alpha", n=n)


def test_execute_search_none_count_uses_default(retrieval_env):
    for i in range(12):
        make_entry(user=retrieval_env, index=i)
    assert len(async_to_sync(execute_search)(retrieval_env, "alpha", n=None)) == 5


def test_nonzero_distance_is_preserved_through_conversion_and_reranking(retrieval_env):
    make_entry(user=retrieval_env)
    make_entry(user=retrieval_env, index=1, embeddings=[1.0, 1.0, 0.0])
    rows = async_to_sync(text_search.query)("alpha", retrieval_env)
    hits = text_search.to_retrieval_hits(rows)
    assert hits[1].cosine_distance == pytest.approx(1 - 1 / math.sqrt(2))
    responses = list(text_search.collate_results(rows))
    reranked = text_search.rerank_and_sort_results(responses, "alpha", True, get_default_search_model().name)
    assert reranked[0].score == pytest.approx(hits[1].cosine_distance)
    assert reranked[0].cross_score == 0.5
    assert reranked[1].score == 0.0
    assert reranked[1].cross_score == 1.0


def test_path_exclusion_and_empty_scope(retrieval_env):
    wanted = make_entry(user=retrieval_env, file_path="allowed/file.org")
    make_entry(user=retrieval_env, file_path="private/file.org")
    rows = EntryAdapters.apply_filters(retrieval_env, '-file:"private/*"')
    assert {row.pk for row in rows} == {wanted.pk}
    assert not EntryAdapters.apply_filters(None, "alpha").exists()


def test_chat_rag_retrieves_user_documents(retrieval_env, monkeypatch):
    from unittest.mock import AsyncMock

    from khoj.database.models import Conversation
    from khoj.routers import helpers

    conversation = Conversation.objects.create(user=retrieval_env)
    for i in range(35):
        make_entry(user=retrieval_env, index=i)
    make_entry(user=UserFactory(), raw="private foreign content")
    # Query generation is an external LLM boundary; exercise the real RAG retrieval below it.
    monkeypatch.setattr(helpers, "extract_questions", AsyncMock(return_value=["alpha"]))

    async def collect_references():
        return [
            event
            async for event in helpers.search_documents(
                q="alpha",
                n=30,
                d=math.inf,
                user=retrieval_env,
                chat_history=[],
                conversation_id=str(conversation.pk),
            )
        ]

    events = async_to_sync(collect_references)()
    references, queries, query = events[-1]
    assert len(references) == 30
    assert queries == ["alpha"]
    assert query == "alpha"
    assert all(ref["compiled"].startswith("alpha content") for ref in references)
    assert all(ref["query"] == "alpha" for ref in references)

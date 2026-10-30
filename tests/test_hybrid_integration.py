"""Native ParadeDB + pgvector recall on a disposable test_ database only."""

import os
from unittest.mock import AsyncMock

import pytest
from asgiref.sync import async_to_sync
from django.db import connection

from khoj.database.models import Agent, EntryDates
from khoj.routers import helpers
from khoj.search_type import hybrid_search
from khoj.search_type.retrieval_types import HybridOptions
from khoj.utils.state import SearchType
from tests import test_retrieval as retrieval_fixtures
from tests.bm25_utils import ensure_bm25_index
from tests.helpers import ChatModelFactory, ConversationFactory, UserFactory
from tests.test_retrieval import make_entry

retrieval_env = retrieval_fixtures.retrieval_env

pytestmark = [
    pytest.mark.bm25_integration,
    pytest.mark.django_db(transaction=True),
    pytest.mark.skipif(os.getenv("KHOJ_TEST_BM25") != "1", reason="requires isolated pg_search DB"),
]


@pytest.fixture(autouse=True)
def real_index():
    assert connection.settings_dict["NAME"].startswith("test_")
    ensure_bm25_index()


def test_native_independent_candidates_and_bm25_context(retrieval_env, monkeypatch):
    monkeypatch.setenv("KHOJ_RETRIEVAL_MODE", "hybrid")
    dense = make_entry(user=retrieval_env, raw="semantic unrelated", compiled="semantic unrelated")
    lexical = make_entry(
        user=retrieval_env,
        raw="raretoken lexical evidence",
        compiled="raretoken lexical evidence",
        embeddings=[0.0, 1.0, 0.0],
    )
    hits = async_to_sync(hybrid_search.query)(
        "raretoken", retrieval_env, question_embedding=[1, 0, 0], max_distance=0.1
    )
    by_id = {hit.entry_id: hit for hit in hits}
    assert set(by_id) == {dense.pk, lexical.pk}
    assert by_id[dense.pk].dense_rank == 1 and by_id[dense.pk].bm25_rank is None
    assert by_id[lexical.pk].bm25_rank == 1 and by_id[lexical.pk].dense_rank is None
    assert by_id[lexical.pk].bm25_score > 0
    conversation = ConversationFactory(user=retrieval_env)
    monkeypatch.setattr(helpers, "extract_questions", AsyncMock(return_value=["raretoken", "lexical"]))

    async def chat():
        return [r async for r in helpers.search_documents("question", 5, 0.1, retrieval_env, [], str(conversation.id))][
            -1
        ]

    refs, queries, _ = async_to_sync(chat)()
    assert "raretoken lexical evidence" in [r["compiled"] for r in refs]
    messages = helpers.build_conversation_context("question", refs, {}, {}, [], model_name="gpt-4o-mini")
    assert "raretoken lexical evidence" in str([m.content for m in messages])


def test_both_scoped_before_top_k_with_common_filters(retrieval_env):
    foreign = UserFactory()
    for i in range(8):
        make_entry(user=foreign, raw="retrieval required", compiled="retrieval required", index=i)
        make_entry(
            user=retrieval_env, raw="retrieval required", compiled="retrieval required", file_type="org", index=i
        )
    allowed = make_entry(
        user=retrieval_env,
        raw="retrieval required",
        compiled="retrieval required",
        file_type="pdf",
        file_path="allowed.pdf",
    )
    blocked = make_entry(
        user=retrieval_env,
        raw="retrieval required blocked",
        compiled="retrieval required blocked",
        file_type="pdf",
        file_path="blocked.pdf",
    )
    old = make_entry(
        user=retrieval_env,
        raw="retrieval required",
        compiled="retrieval required",
        file_type="pdf",
        file_path="old.pdf",
    )
    for entry, date in [(allowed, "2024-03-01"), (allowed, "2024-04-01"), (blocked, "2024-03-01"), (old, "2020-01-01")]:
        EntryDates.objects.create(entry=entry, date=date)
    q = 'dt>="2024-01-01" retrieval +"required" -"blocked" file:"*.pdf" -file:"blocked*" dt<="2024-12-31"'
    hits = async_to_sync(hybrid_search.query)(
        q,
        retrieval_env,
        SearchType.Pdf,
        question_embedding=[1, 0, 0],
        options=HybridOptions(dense_top_k=1, bm25_top_k=1),
    )
    assert [(h.entry_id, h.dense_rank, h.bm25_rank) for h in hits] == [(allowed.pk, 1, 1)]
    assert async_to_sync(helpers.execute_search)(foreign, q, mode="hybrid", t=SearchType.Pdf) == []


def test_agent_scopes_and_revocation_in_all_modes(retrieval_env):
    foreign = UserFactory()
    agent = Agent.objects.create(
        name="Hybrid scope", creator=retrieval_env, privacy_level="private", chat_model=ChatModelFactory()
    )
    own = make_entry(user=retrieval_env, raw="retrieval own", compiled="retrieval own")
    scoped = make_entry(agent=agent, raw="retrieval agent", compiled="retrieval agent")
    make_entry(user=foreign, raw="retrieval forbidden", compiled="retrieval forbidden")
    other = Agent.objects.create(name="Unrelated scope", creator=foreign, chat_model=agent.chat_model)
    make_entry(agent=other, raw="retrieval forbidden agent", compiled="retrieval forbidden agent")
    for mode in ["vector", "bm25", "hybrid"]:
        rows = async_to_sync(helpers.execute_search)(retrieval_env, "retrieval", agent=agent, mode=mode)
        assert {r.entry for r in rows} == {own.raw, scoped.raw}
        assert async_to_sync(helpers.execute_search)(foreign, "retrieval", agent=agent, mode=mode) == []
        assert async_to_sync(helpers.execute_search)(None, "retrieval", agent=agent, mode=mode) == []
    Agent.objects.filter(pk=agent.pk).update(privacy_level="public")
    assert {r.entry for r in async_to_sync(helpers.execute_search)(None, "retrieval", agent=agent, mode="hybrid")} == {
        scoped.raw
    }
    Agent.objects.filter(pk=agent.pk).update(privacy_level="private")
    assert async_to_sync(helpers.execute_search)(None, "retrieval", agent=agent, mode="hybrid") == []


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_private_agent_owner_chat_with_agent_only_document_scope(retrieval_env, monkeypatch, mode):
    from khoj.utils.helpers import ConversationCommand

    monkeypatch.setenv("KHOJ_RETRIEVAL_MODE", mode)
    agent = Agent.objects.create(
        name="Private owner chat", creator=retrieval_env, privacy_level="private", chat_model=ChatModelFactory()
    )
    own = make_entry(
        user=retrieval_env,
        raw="retrieval personal should be excluded",
        compiled="retrieval personal should be excluded",
    )
    scoped = make_entry(
        agent=agent, raw="retrieval private agent evidence", compiled="retrieval private agent evidence"
    )
    conversation = ConversationFactory(user=retrieval_env, agent=agent)
    monkeypatch.setattr(helpers, "extract_questions", AsyncMock(return_value=["retrieval"]))

    async def run():
        return [
            r
            async for r in helpers.search_documents(
                "question",
                5,
                0.1,
                retrieval_env,
                [],
                str(conversation.id),
                conversation_commands=[ConversationCommand.General],
                agent=agent,
            )
        ][-1]

    refs, _, _ = async_to_sync(run)()
    assert [r["compiled"] for r in refs] == [scoped.raw]
    assert own.raw not in str(refs)


def test_direct_hybrid_filter_only_cannot_bypass_private_agent_authorization(retrieval_env):
    foreign = UserFactory()
    agent = Agent.objects.create(
        name="Private filter-only", creator=retrieval_env, privacy_level="private", chat_model=ChatModelFactory()
    )
    make_entry(agent=agent, raw="private evidence", compiled="private evidence")
    with pytest.raises(PermissionError):
        async_to_sync(hybrid_search.query)('file:"notes/*"', foreign, agent=agent, question_embedding=[1, 0, 0])


def test_independent_forty_each_then_fusion_forty_before_rerank(retrieval_env, monkeypatch, caplog):
    import logging

    from khoj.utils import state

    for index in range(45):
        make_entry(user=retrieval_env, index=index, raw=f"semantic only {index}", compiled=f"semantic only {index}")
        make_entry(
            user=retrieval_env,
            index=index + 50,
            raw=f"lexicaltoken only {index}",
            compiled=f"lexicaltoken only {index}",
            embeddings=[0.0, 1.0, 0.0],
        )
    seen = []
    model = next(iter(state.cross_encoder_model.values()))

    def predict(query, rows):
        seen.append(list(rows))
        return [0.5] * len(rows)

    monkeypatch.setattr(model, "predict", predict)
    with caplog.at_level(logging.INFO, logger="khoj.search_type.hybrid_search"):
        rows = async_to_sync(helpers.execute_search)(
            retrieval_env, "lexicaltoken", n=100, r=True, max_distance=0.1, mode="hybrid", dedupe=False
        )
    assert len(rows) == 40 and len(seen[0]) == 40
    assert any(r.additional["dense_rank"] is None for r in rows)
    assert any(r.additional["bm25_rank"] is None for r in rows)
    assert "dense_count=40 bm25_count=40 overlap=0 unique=80 fused=40" in caplog.text


def test_dense_uses_primary_key_to_break_equal_distance_ties(retrieval_env):
    from django.test.utils import CaptureQueriesContext

    from khoj.database.adapters import EntryAdapters

    entries = [make_entry(user=retrieval_env, index=i) for i in range(4)]
    with CaptureQueriesContext(connection) as queries:
        rows = list(EntryAdapters.search_with_embeddings("alpha", [1, 0, 0], retrieval_env, max_results=2))
    assert [r.pk for r in rows] == [r.pk for r in entries[:2]]
    ordering = queries[0]["sql"].split("ORDER BY", 1)[1]
    assert '"database_entry"."id" ASC' in ordering

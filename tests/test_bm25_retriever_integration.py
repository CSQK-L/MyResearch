"""Real ICU/BM25 scores, filtering and plans on disposable PostgreSQL."""

import json
import math
import os
import uuid

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from khoj.database.adapters import EntryAdapters
from khoj.database.lexical_search import ranked_bm25_entries
from khoj.database.models import Agent, EntryDates
from tests.bm25_utils import ensure_bm25_index, make_entry
from tests.helpers import ChatModelFactory

pytestmark = [
    pytest.mark.bm25_integration,
    pytest.mark.django_db(transaction=True),
    pytest.mark.skipif(os.getenv("KHOJ_TEST_BM25") != "1", reason="set KHOJ_TEST_BM25=1 on disposable pg_search DB"),
]


@pytest.fixture(autouse=True)
def bm25_index():
    ensure_bm25_index()


def search(query, user, **kwargs):
    return EntryAdapters.search_with_bm25(query, user, **kwargs)


@pytest.mark.parametrize(
    "query,text",
    [
        ("PostgreSQL", "PostgreSQL vector similarity"),
        ("联邦", "你好，联邦学习中的图神经网络训练"),
        ("BM25 关键词", "BM25 支持关键词检索"),
        ("pgvector", "pgvector indexes"),
        ('"C++" : (PostgreSQL)', "PostgreSQL C++"),
    ],
)
def test_multilingual_and_literal_input(default_user, query, text):
    entry = make_entry(default_user, text)
    assert [hit.entry_id for hit in search(query, default_user)] == [entry.pk]
    assert search("unmatchedlexemezz", default_user) == []


def test_scoring_chunks_stable_order_and_db_limit(default_user, record_property):
    corpus = uuid.uuid4()
    entries = [make_entry(default_user, "keyword retrieval", corpus_id=corpus) for _ in range(6)]
    make_entry(default_user, "keyword " + "irrelevant " * 30)
    with CaptureQueriesContext(connection) as queries:
        hits = search("keyword retrieval", default_user, max_results=4)
    assert len(queries) == 1 and "LIMIT 4" in queries[0]["sql"]
    assert [hit.entry_id for hit in hits] == [entry.pk for entry in entries[:4]]
    assert [hit.bm25_rank for hit in hits] == [1, 2, 3, 4]
    assert all(math.isfinite(hit.bm25_score) and hit.bm25_score > 0 for hit in hits)
    assert [(h.bm25_score, -h.entry_id) for h in hits] == sorted(
        [(h.bm25_score, -h.entry_id) for h in hits], reverse=True
    )
    assert all(h.dense_rank is h.cosine_distance is None and h.rrf_score == 0 for h in hits)
    qs = ranked_bm25_entries(
        EntryAdapters.apply_filters(default_user, "keyword retrieval"), "keyword retrieval", 4
    ).values_list("pk", "bm25_score")
    plan = qs.explain(analyze=True, buffers=True, format="json")
    record_property("bm25_explain", plan)
    parsed = json.loads(plan)
    record_property("bm25_execution_ms", parsed[0]["Execution Time"])
    print("FILTERED BM25 EXPLAIN:", plan)
    assert "ParadeDB" in plan or "khoj_entry_bm25_v1" in plan


def test_permissions_and_filters_before_top_k(default_user, default_user2):
    # Foreign/filtered rows greatly outnumber allowed rows and must not consume K.
    for _ in range(12):
        make_entry(default_user2, "retrieval required")
        make_entry(default_user, "retrieval required", file_path="exclude.org", file_type="org")
    allowed = make_entry(
        default_user, "retrieval required", file_path="allowed.pdf", file_type="pdf", file_source="github"
    )
    blocked = make_entry(default_user, "retrieval required blocked", file_path="blocked.pdf", file_type="pdf")
    EntryDates.objects.bulk_create(
        [
            EntryDates(entry=allowed, date="2024-03-01"),
            EntryDates(entry=allowed, date="2024-04-01"),
            EntryDates(entry=blocked, date="2024-03-01"),
        ]
    )
    query = 'dt>="2024-01-01" retrieval +"required" -"blocked" file:"*.pdf" -file:"blocked*" dt<="2024-12-31"'
    hits = search(query, default_user, max_results=4, file_type_filter="pdf", file_source_filter="github")
    assert [h.entry_id for h in hits] == [allowed.pk]
    assert [h.entry_id for h in search('retrieval -file:"blocked*" file:"*.pdf"', default_user)] == [allowed.pk]
    assert {h.entry_id for h in search("retrieval", default_user2)}.isdisjoint({allowed.pk, blocked.pk})


@pytest.mark.parametrize("visibility,accessible", [("private", False), ("public", True), ("protected", True)])
def test_agent_policy_and_current_visibility(default_user, default_user2, visibility, accessible):
    agent = Agent.objects.create(
        name="bm25 visibility", creator=default_user2, privacy_level=visibility, chat_model=ChatModelFactory()
    )
    own = make_entry(default_user)
    agent_entry = make_entry(agent=agent)
    unrelated = Agent.objects.create(name="unrelated", creator=default_user2, chat_model=agent.chat_model)
    make_entry(agent=unrelated)
    if accessible:
        assert {h.entry_id for h in search("retrieval", default_user, agent=agent)} == {own.pk, agent_entry.pk}
        assert {h.entry_id for h in search("retrieval", None, agent=agent)} == {agent_entry.pk}
        Agent.objects.filter(pk=agent.pk).update(privacy_level="private")
    with pytest.raises(PermissionError):
        search("retrieval", default_user, agent=agent)
    assert {h.entry_id for h in search("retrieval", default_user2, agent=agent)} == {agent_entry.pk}


def test_injection_is_literal_and_table_remains(default_user):
    entry = make_entry(default_user, "retrieval")
    search("'; DROP TABLE database_entry; --", default_user)
    assert [h.entry_id for h in search("retrieval", default_user)] == [entry.pk]


def test_missing_index_reports_database_failure(default_user):
    from django.db import DatabaseError, transaction

    # Shadow the real table only in this session. No existing index is touched.
    # Keeping the table name also preserves Django's cached column aliases.
    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute("CREATE TEMP TABLE database_entry (LIKE database_entry INCLUDING DEFAULTS) ON COMMIT DROP")
        with pytest.raises(DatabaseError, match="index"):
            with transaction.atomic():
                search("retrieval", default_user)


def test_tokenless_input_does_not_match_every_document(default_user):
    make_entry(default_user, "retrieval")
    assert search('!!! () : ""', default_user) == []

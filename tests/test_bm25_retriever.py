"""BM25 contract and generated SQL tests; pg_search is not needed here."""

from unittest.mock import patch

import pytest
from django.db import ProgrammingError, connection
from django.db.models.query import QuerySet

from khoj.database.adapters import EntryAdapters
from khoj.search_type.retrieval_types import RetrievalHit


@pytest.mark.parametrize("query", ["", "   ", 'file:"*.pdf"', '-file:"*.pdf"', '+"required"', 'dt:"2024-01-01"'])
def test_empty_terms_do_not_query(query, default_user, django_assert_num_queries):
    with django_assert_num_queries(0):
        assert EntryAdapters.search_with_bm25(query, default_user) == []


@pytest.mark.parametrize("limit", [-1, 1.5, "4", None, True, False])
def test_invalid_limits(limit, default_user):
    with pytest.raises(ValueError, match="max_results"):
        EntryAdapters.search_with_bm25("retrieval", default_user, max_results=limit)


def test_zero_and_no_scope_do_not_query(default_user, django_assert_num_queries):
    with django_assert_num_queries(0):
        assert EntryAdapters.search_with_bm25("retrieval", default_user, max_results=0) == []
        assert EntryAdapters.search_with_bm25("retrieval", None) == []


def test_returns_existing_contract(default_user):
    # Replace only DB evaluation, keeping expression/QuerySet construction real.
    with patch.object(QuerySet, "_fetch_all", lambda qs: setattr(qs, "_result_cache", [(101, 8.3), (205, 6.1)])):
        hits = EntryAdapters.search_with_bm25("BM25", default_user, max_results=2)
    assert hits == [RetrievalHit(101, bm25_rank=1, bm25_score=8.3), RetrievalHit(205, bm25_rank=2, bm25_score=6.1)]


@pytest.mark.parametrize("term", ["你好 BM25", '"quoted" C++ : ()', "x'; DROP TABLE database_entry; --"])
def test_parameterized_query_filters_before_limit(term, default_user):
    captured = []

    def fetch(qs):
        captured.append(qs.query.sql_with_params())
        qs._result_cache = []

    with patch.object(QuerySet, "_fetch_all", fetch):
        EntryAdapters.search_with_bm25(
            term + ' file:"*.pdf" -"blocked"', default_user, max_results=3, file_type_filter="pdf"
        )
    sql, params = captured[0]
    assert "||| %s" in sql and "pdb.score(" in sql
    assert term not in sql and term in params
    assert default_user.pk in params and "pdf" in params
    assert "%blocked%" in params and r".*\.pdf" in params
    assert "ORDER BY" in sql and "DESC" in sql and '"id" ASC' in sql and sql.endswith("LIMIT 3")


def test_database_failure_is_not_silenced(default_user):
    with patch.object(
        QuerySet, "_fetch_all", side_effect=ProgrammingError("operator does not exist: text ||| unknown")
    ):
        with pytest.raises(ProgrammingError, match="operator does not exist"):
            EntryAdapters.search_with_bm25("keyword", default_user)


def test_missing_extension_real_database(default_user):
    with connection.cursor() as cursor:
        cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname='pg_search')")
        installed = cursor.fetchone()[0]
    if installed:
        pytest.skip("no-extension contract is exercised on standard PostgreSQL")
    with pytest.raises(ProgrammingError):
        EntryAdapters.search_with_bm25("keyword", default_user)


def test_private_agent_requires_existing_authorization(default_user, default_user2):
    from khoj.database.models import Agent
    from tests.helpers import ChatModelFactory

    agent = Agent.objects.create(name="private-bm25", creator=default_user2, chat_model=ChatModelFactory())
    with pytest.raises(PermissionError):
        EntryAdapters.search_with_bm25("keyword", default_user, agent=agent)


@pytest.mark.parametrize("query", [None, 3, [], {}])
def test_invalid_query_type(query, default_user):
    with pytest.raises(ValueError, match="raw_query"):
        EntryAdapters.search_with_bm25(query, default_user)

"""API validation and mode propagation through real search execution."""

import pytest
from django.db import OperationalError

from khoj.database.adapters import EntryAdapters
from tests import test_hybrid_retrieval as hybrid_fixtures
from tests import test_retrieval as client_fixtures
from tests import test_retrieval as retrieval_fixtures

lexical_only = hybrid_fixtures.lexical_only
vector_default = hybrid_fixtures.vector_default
retrieval_client = client_fixtures.retrieval_client
retrieval_env = retrieval_fixtures.retrieval_env

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("vector", ["semantic alpha"]),
        ("bm25", ["lexical evidence"]),
        ("hybrid", ["semantic alpha", "lexical evidence"]),
    ],
)
def test_api_modes(retrieval_client, lexical_only, mode, expected):
    response = retrieval_client.get("/api/search", params={"q": "alpha", "mode": mode, "max_distance": 0.1})
    assert response.status_code == 200, response.text
    assert [r["entry"] for r in response.json()] == expected
    for row in response.json():
        assert row["score"] == (1.0 if row["entry"] == "lexical evidence" else 0.0)


def test_invalid_mode_and_server_default(retrieval_client, lexical_only, monkeypatch):
    assert retrieval_client.get("/api/search", params={"q": "alpha", "mode": "wrong"}).status_code == 422
    monkeypatch.setenv("KHOJ_RETRIEVAL_MODE", "hybrid")
    response = retrieval_client.get("/api/search", params={"q": "alpha", "max_distance": 0.1})
    assert len(response.json()) == 2
    response = retrieval_client.get("/api/search", params={"q": "alpha", "max_distance": 0.1, "mode": "vector"})
    assert len(response.json()) == 1


def test_bm25_unavailable_is_explicit_and_hybrid_degrades(retrieval_client, lexical_only, monkeypatch):
    def broken(**kwargs):
        raise OperationalError("missing pg_search/index")

    monkeypatch.setattr(EntryAdapters, "search_with_bm25", broken)
    response = retrieval_client.get("/api/search", params={"q": "alpha", "mode": "bm25"})
    assert response.status_code == 503
    assert response.json()["detail"] == "BM25 backend unavailable"
    response = retrieval_client.get("/api/search", params={"q": "alpha", "mode": "hybrid", "max_distance": 0.1})
    assert response.status_code == 200
    assert [r["entry"] for r in response.json()] == ["semantic alpha"]

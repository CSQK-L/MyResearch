"""Real pg_search tests. Opt in explicitly on a disposable, preconfigured test DB."""

import json
import math
import os
import uuid
from io import StringIO

import pytest
from django.core.management import call_command
from django.db import connection, transaction

from khoj.database.management.commands.setup_bm25_index import (
    INDEX_NAME,
    index_status,
    inspect_database,
    prerequisites,
    quote_identifier,
)
from khoj.database.models import Entry

pytestmark = [
    pytest.mark.bm25_integration,
    pytest.mark.django_db(transaction=True),
    pytest.mark.skipif(
        os.getenv("KHOJ_TEST_BM25") != "1", reason="set KHOJ_TEST_BM25=1 on a disposable pg_search 0.26.x test DB"
    ),
]


def test_real_bm25_index_lifecycle(default_user, record_property):
    environment = inspect_database(connection)
    assert str(connection.settings_dict["NAME"]).startswith("test_"), "Use a disposable pytest test_ database"
    assert not prerequisites(environment), prerequisites(environment)
    assert index_status(environment)[0] == "missing", "Use a fresh test DB; existing indexes are never dropped"
    texts = [
        "PostgreSQL supports vector similarity search.",
        "BM25 is useful for keyword retrieval.",
        "Hybrid retrieval combines semantic and lexical search.",
        "你好，联邦学习中的图神经网络训练",
        "Hybrid retrieval combines BM25 and embeddings.",
        "PostgreSQL 支持向量检索和关键词匹配。",
    ]
    entries = Entry.objects.bulk_create(
        [
            Entry(
                user=default_user,
                embeddings=[1.0, 0.0, 0.0],
                raw=text,
                compiled=text,
                heading=f"Document {i}",
                hashed_value=str(uuid.uuid4()),
                file_path=f"bm25/{i}.txt",
            )
            for i, text in enumerate(texts)
        ]
    )

    def snapshot():
        return [
            dict(row, embeddings=tuple(row["embeddings"]))
            for row in Entry.objects.order_by("pk").values("pk", "compiled", "raw", "embeddings", "user_id", "agent_id")
        ]

    before = snapshot()
    output = StringIO()
    call_command("setup_bm25_index", "--create", stdout=output)
    assert index_status(inspect_database(connection))[0] == "ready"
    assert snapshot() == before
    quote = quote_identifier
    table = f"{quote(environment.schema)}.{quote(environment.table)}"
    pk, compiled = quote(Entry._meta.pk.column), quote(Entry._meta.get_field("compiled").column)
    query = f"SELECT {pk}, pdb.score({pk}) AS bm25_score FROM {table} WHERE {compiled} ||| %s ORDER BY bm25_score DESC, {pk} LIMIT 5"

    def search(text):
        with connection.cursor() as cursor:
            cursor.execute(query, [text])
            return cursor.fetchall()

    rows = search("keyword retrieval")
    assert rows and rows[0][0] == entries[1].pk
    assert all(math.isfinite(score) and score >= 0 for _, score in rows)
    assert [score for _, score in rows] == sorted((score for _, score in rows), reverse=True)
    with connection.cursor() as cursor:
        cursor.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + query, ["keyword retrieval"])
        plan = cursor.fetchone()[0]
        for sample in [texts[0], texts[3], texts[5]]:
            cursor.execute("SELECT %s::pdb.icu::text[]", [sample])
            assert cursor.fetchone()[0], f"No ICU tokens for {sample}"
    plan_text = json.dumps(plan, ensure_ascii=False)
    record_property("bm25_explain", plan_text)
    record_property("bm25_index_observed", INDEX_NAME in plan_text or "paradedb" in plan_text.lower())
    record_property("bm25_execution_ms", plan[0]["Execution Time"])
    print("BM25 EXPLAIN (inspect access path, buffers, sort and timing):", plan_text)
    # Tiny tables may use a different plan; record evidence rather than force planner settings.
    assert plan[0]["Plan"]["Actual Rows"] > 0
    for term, expected in [("联邦", entries[3].pk), ("embeddings", entries[4].pk), ("关键词", entries[5].pk)]:
        assert expected in {pk for pk, _ in search(term)}

    with connection.cursor() as cursor:
        cursor.execute("SHOW search_path")
        original_path = cursor.fetchone()[0]
        try:
            for path in [f"pg_catalog, {quote(environment.schema)}", f'pg_catalog, "pdb", {quote(environment.schema)}']:
                cursor.execute("SELECT set_config('search_path', %s, false)", [path])
                assert index_status(inspect_database(connection))[0] == "ready"
                call_command("setup_bm25_index", "--create", stdout=StringIO())
        finally:
            cursor.execute("SELECT set_config('search_path', %s, false)", [original_path])

    original_oid = None
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass(%s)::oid", [f"{quote(environment.schema)}.{quote(INDEX_NAME)}"])
        original_oid = cursor.fetchone()[0]
    call_command("setup_bm25_index", "--create", stdout=StringIO())
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass(%s)::oid", [f"{quote(environment.schema)}.{quote(INDEX_NAME)}"])
        assert cursor.fetchone()[0] == original_oid

    added = Entry.objects.bulk_create(
        [
            Entry(
                user=default_user,
                embeddings=[1.0, 0.0, 0.0],
                raw="oldlexemezz",
                compiled="oldlexemezz",
                hashed_value=str(uuid.uuid4()),
            )
        ]
    )[0]
    assert {pk for pk, _ in search("oldlexemezz")} == {added.pk}
    Entry.objects.filter(pk=added.pk).update(compiled="newlexemezz")
    assert search("oldlexemezz") == []
    assert {pk for pk, _ in search("newlexemezz")} == {added.pk}
    Entry.objects.filter(pk=added.pk).delete()
    assert search("newlexemezz") == []
    with pytest.raises(RuntimeError, match="rollback"):
        with transaction.atomic():
            Entry.objects.bulk_create(
                [
                    Entry(
                        user=default_user,
                        embeddings=[1.0, 0.0, 0.0],
                        raw="rollbacklexemezz",
                        compiled="rollbacklexemezz",
                        hashed_value=str(uuid.uuid4()),
                    )
                ]
            )
            Entry.objects.filter(pk=entries[0].pk).update(compiled="rollbackchangezz")
            Entry.objects.filter(pk=entries[2].pk).delete()
            raise RuntimeError("rollback")
    assert search("rollbacklexemezz") == []
    assert search("rollbackchangezz") == []
    assert entries[0].pk in {pk for pk, _ in search("similarity")}
    assert entries[2].pk in {pk for pk, _ in search("lexical")}
    assert index_status(inspect_database(connection))[0] == "ready"

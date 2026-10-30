"""Helpers shared only by opt-in BM25 tests."""

import uuid
from io import StringIO

from django.core.management import call_command
from django.db import connection

from khoj.database.management.commands.setup_bm25_index import index_status, inspect_database, prerequisites
from khoj.database.models import Entry


def ensure_bm25_index():
    assert str(connection.settings_dict["NAME"]).startswith("test_"), "Use a disposable test_ database"
    environment = inspect_database(connection)
    assert not prerequisites(environment), prerequisites(environment)
    status, details = index_status(environment)
    assert status in ("missing", "ready"), details
    if status == "missing":
        call_command("setup_bm25_index", "--create", stdout=StringIO())
    assert index_status(inspect_database(connection))[0] == "ready"


def make_entry(user=None, text="retrieval", **kwargs):
    values = dict(
        user=user,
        raw=text,
        compiled=text,
        embeddings=[1.0, 0.0, 0.0],
        hashed_value=str(uuid.uuid4()),
        file_path="notes.txt",
    )
    values.update(kwargs)
    # Entry.save currently omits super().save(); ingestion uses bulk_create.
    return Entry.objects.bulk_create([Entry(**values)])[0]

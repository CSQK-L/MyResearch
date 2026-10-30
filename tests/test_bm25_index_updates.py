"""Exercise Khoj writers with the real transactional ParadeDB index."""

import logging
import os
from types import SimpleNamespace

import pytest
from django.db import connection, transaction

from khoj.database.adapters import AgentAdapters, EntryAdapters, get_default_search_model
from khoj.database.models import Entry
from khoj.processor.content.plaintext.plaintext_to_entries import PlaintextToEntries
from khoj.utils import state
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


def ids(user, term, **kwargs):
    return {h.entry_id for h in EntryAdapters.search_with_bm25(term, user, **kwargs)}


def test_insert_bulk_update_delete_and_bulk_delete(default_user):
    # Ingestion intentionally uses bulk_create; save() is not its write path.
    entry = make_entry(default_user, "oldlexemezz")
    others = Entry.objects.bulk_create(
        [
            Entry(
                user=default_user,
                raw="bulklexemezz",
                compiled="bulklexemezz",
                embeddings=[1.0, 0.0, 0.0],
                hashed_value=str(i),
                file_path="bulk.txt",
            )
            for i in range(3)
        ]
    )
    assert ids(default_user, "oldlexemezz") == {entry.pk}
    assert ids(default_user, "bulklexemezz") == {e.pk for e in others}
    Entry.objects.filter(pk=entry.pk).update(compiled="newlexemezz")
    assert ids(default_user, "oldlexemezz") == set()
    assert ids(default_user, "newlexemezz") == {entry.pk}
    for e in others:
        e.compiled = "changedbulkzz"
    Entry.objects.bulk_update(others, ["compiled"])
    assert ids(default_user, "bulklexemezz") == set()
    assert ids(default_user, "changedbulkzz") == {e.pk for e in others}
    Entry.objects.filter(pk=entry.pk).delete()
    assert ids(default_user, "newlexemezz") == set()
    EntryAdapters.delete_entry_by_file(default_user, "bulk.txt")
    assert ids(default_user, "changedbulkzz") == set()


@pytest.fixture
def processor(monkeypatch):
    model = get_default_search_model()
    # Only embedding inference is stubbed. Parsing/chunking/writing stays real.
    monkeypatch.setattr(
        state,
        "embeddings_model",
        {model.name: SimpleNamespace(embed_documents=lambda texts: [[1.0, 0.0, 0.0] for text in texts])},
    )
    return PlaintextToEntries()


def test_real_document_update_regenerate_and_delete(default_user, processor):
    path = "lifecycle.txt"
    processor.process({path: "oldcontentzz"}, default_user)
    old_ids = ids(default_user, "oldcontentzz")
    assert len(old_ids) == 1
    processor.process({path: "newcontentzz BM25 ParadeDB"}, default_user)
    new_ids = ids(default_user, "newcontentzz")
    assert len(new_ids) == 1 and old_ids.isdisjoint(new_ids)
    assert ids(default_user, "oldcontentzz") == set()
    assert ids(default_user, "BM25") == ids(default_user, "ParadeDB") == new_ids
    processor.process({path: "newcontentzz BM25 ParadeDB"}, default_user)
    assert ids(default_user, "newcontentzz") == new_ids
    processor.process({path: "newcontentzz BM25 ParadeDB"}, default_user, regenerate=True)
    regenerated = ids(default_user, "newcontentzz")
    assert len(regenerated) == 1 and regenerated.isdisjoint(new_ids)
    assert not Entry.objects.filter(pk__in=old_ids | new_ids).exists()
    processor.process({path: ""}, default_user)
    assert ids(default_user, "newcontentzz") == set()
    assert not Entry.objects.filter(user=default_user, file_path=path).exists()


def test_real_split_chunks_keep_individual_pks(default_user, processor):
    from khoj.processor.content.text_to_entries import TextToEntries
    from khoj.utils.rawconfig import Entry as TextEntry

    chunks = TextToEntries.split_entries_by_max_tokens(
        [
            TextEntry(
                raw="chunktermzz alpha beta gamma delta",
                compiled="chunktermzz alpha beta gamma delta",
                file="chunks.txt",
            )
        ],
        max_tokens=2,
    )
    assert len(chunks) > 1 and len({c.corpus_id for c in chunks}) == 1
    processor.update_embeddings(default_user, chunks, "plaintext", "computer", logger=logging.getLogger(__name__))
    rows = list(Entry.objects.filter(user=default_user))
    hits = EntryAdapters.search_with_bm25("chunktermzz alpha beta gamma delta", default_user)
    assert {h.entry_id for h in hits} == {e.pk for e in rows}
    assert len(hits) == len(chunks)
    assert EntryAdapters.delete_all_entries(default_user, batch_size=1) >= len(chunks)
    assert ids(default_user, "chunktermzz alpha beta gamma delta") == set()


def test_actual_agent_copy_update_and_cascade_delete(default_user, default_user2, processor):
    processor.process({"agent.txt": "agentcopyzz"}, default_user)
    source_ids = ids(default_user, "agentcopyzz")
    chat_model = ChatModelFactory()

    def update(files, slug=None):
        return AgentAdapters.atomic_update_agent(
            user=default_user,
            name="bm25 copies",
            personality="helper",
            privacy_level="public",
            icon="Lightbulb",
            color="orange",
            chat_model_option=chat_model,
            files=files,
            input_tools=[],
            output_modes=[],
            slug=slug,
        )

    agent = update(["agent.txt"])
    copied = ids(default_user2, "agentcopyzz", agent=agent)
    assert len(copied) == 1 and copied.isdisjoint(source_ids)
    assert ids(default_user2, "agentcopyzz") == set()
    update([], agent.slug)
    assert ids(default_user2, "agentcopyzz", agent=agent) == set()
    agent = update(["agent.txt"], agent.slug)
    new_copy = ids(default_user2, "agentcopyzz", agent=agent)
    assert len(new_copy) == 1 and new_copy.isdisjoint(copied)
    agent.delete()
    assert not Entry.objects.filter(pk__in=new_copy).exists()
    assert ids(default_user2, "agentcopyzz") == set()
    assert ids(default_user, "agentcopyzz") == source_ids


def test_commit_rollback_and_independent_session_visibility(default_user):
    original = make_entry(default_user, "originalzz")
    deleted = make_entry(default_user, "deletezz")
    observer = connection.copy(alias="bm25_observer")

    def observed(term):
        with observer.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM database_entry WHERE compiled ||| %s AND user_id=%s", [term, default_user.pk]
            )
            return {row[0] for row in cursor.fetchall()}

    try:
        with pytest.raises(RuntimeError, match="rollback"):
            with transaction.atomic():
                added = make_entry(default_user, "uncommittedzz")
                Entry.objects.filter(pk=original.pk).update(compiled="updatedzz")
                Entry.objects.filter(pk=deleted.pk).delete()
                assert ids(default_user, "uncommittedzz") == {added.pk}
                assert ids(default_user, "updatedzz") == {original.pk}
                assert ids(default_user, "deletezz") == set()
                assert observed("uncommittedzz") == observed("updatedzz") == set()
                assert observed("originalzz") == {original.pk}
                assert observed("deletezz") == {deleted.pk}
                raise RuntimeError("rollback")
        assert ids(default_user, "uncommittedzz updatedzz") == set()
        assert ids(default_user, "originalzz") == {original.pk}
        assert ids(default_user, "deletezz") == {deleted.pk}
        with transaction.atomic():
            committed = make_entry(default_user, "committedzz")
            Entry.objects.filter(pk=original.pk).update(compiled="committedupdatezz")
            Entry.objects.filter(pk=deleted.pk).delete()
        assert observed("committedzz") == ids(default_user, "committedzz") == {committed.pk}
        assert observed("committedupdatezz") == {original.pk}
        assert observed("originalzz deletezz") == set()
    finally:
        observer.close()

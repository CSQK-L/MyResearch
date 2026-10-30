"""Chat retrieval through real references and model message construction, offline."""

import json
from unittest.mock import AsyncMock

import pytest
from asgiref.sync import async_to_sync

from khoj.database.adapters import ConversationAdapters, EntryAdapters
from khoj.database.models import EntryDates
from khoj.routers import helpers
from khoj.utils import state
from khoj.utils.helpers import ConversationCommand
from tests import test_hybrid_retrieval as hybrid_fixtures
from tests import test_retrieval as retrieval_fixtures
from tests.helpers import AiModelApiFactory, ChatModelFactory, ConversationFactory

lexical_only = hybrid_fixtures.lexical_only
vector_default = hybrid_fixtures.vector_default
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
def test_inferred_queries_references_and_llm_context(retrieval_env, lexical_only, monkeypatch, mode, expected):
    monkeypatch.setenv("KHOJ_RETRIEVAL_MODE", mode)
    conversation = ConversationFactory(user=retrieval_env)
    monkeypatch.setattr(helpers, "extract_questions", AsyncMock(return_value=["alpha", "second query", "alpha"]))
    model = next(iter(state.cross_encoder_model.values()))
    monkeypatch.setattr(model, "predict", lambda q, rows: [0.5] * len(rows))
    seen = []
    original = EntryAdapters.search_with_bm25

    def lexical(**kwargs):
        seen.append(kwargs["raw_query"])
        return original(**kwargs)

    monkeypatch.setattr(EntryAdapters, "search_with_bm25", lexical)
    for row in lexical_only:
        EntryDates.objects.create(entry=row, date="2024-03-01")
    query = 'dt>="2024-01-01" alpha -file:"secret*" file:"notes/*"'

    async def retrieve():
        return [
            item async for item in helpers.search_documents(query, 5, 0.1, retrieval_env, [], str(conversation.id))
        ][-1]

    refs, inferred, clean = async_to_sync(retrieve)()
    assert inferred == ["alpha", "second query"]
    assert clean == "alpha"
    assert [ref["compiled"] for ref in refs] == expected
    if mode != "vector":
        assert len(seen) == 2
        for raw in seen:
            assert 'dt>="2024-01-01"' in raw and '-file:"secret*"' in raw and 'file:"notes/*"' in raw
    # Use the actual chat model message builder and generation dispatcher.
    chat_model = ChatModelFactory(
        name="gpt-4o-mini",
        model_type="openai",
        ai_model_api=AiModelApiFactory(api_key="offline-test", api_base_url="http://localhost/unused"),
    )
    monkeypatch.setattr(ConversationAdapters, "aget_valid_chat_model", AsyncMock(return_value=chat_model))
    monkeypatch.setattr(ConversationAdapters, "aget_max_context_size", AsyncMock(return_value=4096))
    sent = []

    def provider(messages, **kwargs):
        sent.extend(messages)

        async def response():
            yield helpers.ResponseWithThought(text="offline answer")

        return response()

    monkeypatch.setattr(helpers, "converse_openai", provider)

    async def respond():
        stream, metadata = await helpers.agenerate_chat_response(
            "question", [], conversation, compiled_references=refs, user=retrieval_env
        )
        return [item.text async for item in stream]

    assert async_to_sync(respond)() == ["offline answer"]
    context = json.dumps([message.content for message in sent], ensure_ascii=False)
    for evidence in expected:
        assert evidence in context
    if mode == "vector":
        assert "lexical evidence" not in context


def test_non_document_chat_does_not_retrieve(retrieval_env, monkeypatch):
    monkeypatch.setenv("KHOJ_RETRIEVAL_MODE", "hybrid")

    def forbidden(*args, **kwargs):
        raise AssertionError("retrieval must not run")

    monkeypatch.setattr(EntryAdapters, "search_with_bm25", forbidden)

    async def run():
        return [
            item
            async for item in helpers.search_documents(
                "hello", 5, 0.1, retrieval_env, [], "unused", conversation_commands=[ConversationCommand.General]
            )
        ]

    assert async_to_sync(run)() == [([], [], "hello")]


@pytest.mark.parametrize("bm25_available", [True, False])
@pytest.mark.parametrize("transport", ["post", "websocket"])
@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_actual_chat_transports_use_selected_retrieval(
    retrieval_env, lexical_only, monkeypatch, transport, mode, bm25_available
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from starlette.authentication import AuthCredentials, AuthenticationBackend, SimpleUser
    from starlette.middleware.authentication import AuthenticationMiddleware

    from khoj.database.adapters import AgentAdapters
    from khoj.routers import api_chat

    monkeypatch.setenv("KHOJ_RETRIEVAL_MODE", mode)
    if not bm25_available:
        from django.db import OperationalError

        def unavailable(**kwargs):
            raise OperationalError("BM25 backend unavailable")

        monkeypatch.setattr(EntryAdapters, "search_with_bm25", unavailable)

    conversation = ConversationFactory(user=retrieval_env)

    class Auth(AuthenticationBackend):
        async def authenticate(self, conn):
            user = SimpleUser("chat-test")
            user.object = retrieval_env
            user.client_app = None
            return AuthCredentials(["authenticated"]), user

    app = FastAPI()
    app.include_router(api_chat.api_chat, prefix="/api/chat")
    app.add_middleware(AuthenticationMiddleware, backend=Auth())
    monkeypatch.setattr(api_chat, "ALLOWED_HOSTS", ["testserver"])
    monkeypatch.setattr(api_chat, "update_telemetry_state", lambda **kwargs: None)
    monkeypatch.setattr(api_chat, "save_to_conversation_log", AsyncMock())
    monkeypatch.setattr(api_chat, "is_ready_to_chat", AsyncMock(return_value=True))
    monkeypatch.setattr(AgentAdapters, "aget_default_agent", AsyncMock(return_value=None))
    monkeypatch.setattr(ConversationAdapters, "ais_memory_enabled", AsyncMock(return_value=False))
    monkeypatch.setattr(
        api_chat,
        "aget_data_sources_and_output_format",
        AsyncMock(return_value={"sources": [ConversationCommand.Notes], "output": ConversationCommand.Text}),
    )
    monkeypatch.setattr(helpers, "extract_questions", AsyncMock(return_value=["alpha", "second query"]))
    chat_model = ChatModelFactory(
        name="gpt-4o-mini",
        model_type="openai",
        ai_model_api=AiModelApiFactory(api_key="offline", api_base_url="http://localhost/unused"),
    )
    monkeypatch.setattr(ConversationAdapters, "aget_valid_chat_model", AsyncMock(return_value=chat_model))
    monkeypatch.setattr(ConversationAdapters, "aget_max_context_size", AsyncMock(return_value=4096))
    sent = []

    def provider(messages, **kwargs):
        sent.extend(messages)

        async def response():
            yield helpers.ResponseWithThought(text="offline chat answer")

        return response()

    monkeypatch.setattr(helpers, "converse_openai", provider)
    body = {"q": "question", "n": 5, "d": 0.1, "conversation_id": str(conversation.id), "stream": False}
    with TestClient(app) as client:
        if transport == "post":
            response = client.post("/api/chat", json=body)
            assert response.status_code == 200, response.text
            payload = response.json()
        else:
            events = []
            with client.websocket_connect("/api/chat/ws", headers={"origin": "http://testserver"}) as ws:
                ws.send_json(body)
                for _ in range(80):
                    event = ws.receive_text()
                    events.append(event)
                    if event.startswith("{") and json.loads(event).get("type") == "end_response":
                        break
                else:
                    pytest.fail("WebSocket did not finish")
            payload = events
    serialized = json.dumps(payload)
    context = str([message.content for message in sent])
    assert "offline chat answer" in serialized
    assert "semantic alpha" in serialized and "semantic alpha" in context
    assert ("lexical evidence" in serialized) == (mode == "hybrid" and bm25_available)
    assert ("lexical evidence" in context) == (mode == "hybrid" and bm25_available)

"""Phase one: no real LLM requests; exercise the existing SDK and ORM boundaries."""

import json
from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from fastapi import HTTPException
from langchain_core.messages.chat import ChatMessage

from khoj.database.adapters import AgentAdapters, ConversationAdapters
from khoj.database.models import AiModelApi, ChatModel, PriceTier
from khoj.processor.conversation.openai import gpt, utils
from khoj.processor.conversation.utils import ResponseWithThought
from khoj.routers import helpers

KEY = "test-only-deepseek-secret"
BASE = "https://proxy.example.test/deepseek/v1"
MESSAGES = [ChatMessage(role="user", content="你好")]


@pytest.fixture
def deepseek_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
    monkeypatch.setenv("DEEPSEEK_BASE_URL", BASE)


def register():
    output = StringIO()
    call_command("register_deepseek_models", stdout=output)
    return output.getvalue()


def test_registration_is_isolated_idempotent(deepseek_env, caplog):
    original_api = AiModelApi.objects.create(name="OpenAI", api_key="old", api_base_url=None)
    original_model = ChatModel.objects.create(name="gpt-4o", ai_model_api=original_api, model_type="openai")
    before_default = ConversationAdapters.get_default_chat_model()
    output = register() + register()
    models = ChatModel.objects.filter(model_type="deepseek").order_by("name")
    assert list(models.values_list("name", flat=True)) == ["deepseek-flash", "deepseek-v4-pro"]
    assert AiModelApi.objects.filter(name="DeepSeek").count() == 1
    for model in models:
        model.full_clean()
        assert model.ai_model_api.api_key == KEY
        assert model.ai_model_api.api_base_url == BASE
        assert model.price_tier == PriceTier.STANDARD
        assert not model.vision_enabled
        assert model.max_prompt_size == 60000
    original_api.refresh_from_db()
    original_model.refresh_from_db()
    assert original_api.api_key == "old" and original_api.api_base_url is None
    assert original_model.model_type == "openai"
    assert ConversationAdapters.get_default_chat_model() == before_default
    assert KEY not in output + caplog.text


def test_missing_key_creates_nothing(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(CommandError, match="DEEPSEEK_API_KEY"):
        register()
    assert not AiModelApi.objects.exists()
    assert not ChatModel.objects.exists()


@pytest.mark.parametrize("conflict", ["key", "base", "provider_type", "duplicate", "model"])
def test_registration_conflicts_fail_without_partial_writes(deepseek_env, conflict):
    api = AiModelApi.objects.create(name="DeepSeek", api_key=KEY, api_base_url=BASE)
    if conflict == "key":
        api.api_key = "existing-secret"
        api.save()
    elif conflict == "base":
        api.api_base_url = "https://different.example.test"
        api.save()
    elif conflict == "provider_type":
        ChatModel.objects.create(name="custom", model_type="openai", ai_model_api=api)
    elif conflict == "duplicate":
        AiModelApi.objects.create(name="DeepSeek", api_key=KEY, api_base_url=BASE)
    else:
        ChatModel.objects.create(name="deepseek-v4-pro", model_type="deepseek", ai_model_api=None)
    before = list(ChatModel.objects.values())
    with pytest.raises(CommandError) as error:
        register()
    assert KEY not in str(error.value)
    assert "existing-secret" not in str(error.value)
    assert list(ChatModel.objects.values()) == before


@pytest.mark.parametrize("models", [["custom", "gpt-4o-mini"], ["gpt-4o-mini", "custom"]])
def test_vision_does_not_depend_on_model_order(models):
    from khoj.utils.initialization import _setup_chat_model_provider

    _setup_chat_model_provider(ChatModel.ModelType.OPENAI, models, "fake-key", False, vision_enabled=True)
    assert ChatModel.objects.get(name="gpt-4o-mini").vision_enabled
    assert not ChatModel.objects.get(name="custom").vision_enabled


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["deepseek", "openai", "anthropic", "google"])
@pytest.mark.parametrize("valid", [True, False])
async def test_validation(provider, valid):
    model = ChatModel(
        name="custom", model_type=provider, ai_model_api=AiModelApi(api_key=KEY, api_base_url=BASE) if valid else None
    )
    conversation = SimpleNamespace(agent=None)
    with (
        patch.object(ConversationAdapters, "aget_user_chat_model", AsyncMock(return_value=model)),
        patch.object(ConversationAdapters, "aget_chat_model", AsyncMock(return_value=model)),
        patch.object(AgentAdapters, "aget_default_agent", AsyncMock(return_value=None)),
    ):
        if valid:
            assert await helpers.is_ready_to_chat(None)
            assert await ConversationAdapters.aget_valid_chat_model(None, conversation, True) is model
        else:
            with pytest.raises(HTTPException):
                await helpers.is_ready_to_chat(None)
            with pytest.raises(ValueError):
                await ConversationAdapters.aget_valid_chat_model(None, conversation, True)


def test_sync_validation_rejects_unconfigured_deepseek():
    with patch.object(ConversationAdapters, "get_default_chat_model", return_value=ChatModel(model_type="deepseek")):
        with pytest.raises(HTTPException):
            helpers.validate_chat_model(None)


def sdk_response(stream):
    if stream:
        chunks = [
            dict(
                id="test",
                object="chat.completion.chunk",
                created=1,
                model="custom",
                choices=[dict(index=0, delta={"content": text}, finish_reason=None)],
            )
            for text in ["你好", "，我是 DeepSeek。"]
        ]
        chunks.append(
            dict(
                id="test",
                object="chat.completion.chunk",
                created=1,
                model="custom",
                choices=[],
                usage=dict(prompt_tokens=3, completion_tokens=8, total_tokens=11),
            )
        )
        content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})
    return httpx.Response(
        200,
        json=dict(
            id="test",
            object="chat.completion",
            created=1,
            model="custom",
            choices=[
                dict(index=0, message=dict(role="assistant", content="你好，我是 DeepSeek。"), finish_reason="stop")
            ],
            usage=dict(prompt_tokens=3, completion_tokens=8, total_tokens=11),
        ),
    )


@pytest.mark.parametrize("provider", ["deepseek", "openai"])
@pytest.mark.parametrize("stream", [True, False])
def test_sync_sdk_parameters(provider, stream):
    requests = []

    def handler(request):
        requests.append(request)
        return sdk_response(stream)

    client = openai.OpenAI(api_key=KEY, base_url=BASE, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    with (
        patch.object(utils, "get_openai_client", return_value=client),
        patch.dict(utils.openai_clients, {}, clear=True),
        patch.object(utils, "is_non_streaming_model", return_value=not stream),
    ):
        result = gpt.openai_send_message_to_model(MESSAGES, KEY, "custom", api_base_url=BASE, provider_type=provider)
    assert isinstance(result, ResponseWithThought)
    assert result.text == "你好，我是 DeepSeek。"
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == BASE + "/chat/completions"
    assert request.headers["authorization"] == "Bearer " + KEY
    body = json.loads(request.content)
    assert body["model"] == "custom"
    if provider == "deepseek":
        assert body["max_tokens"] == 16000
        assert "max_completion_tokens" not in body
        assert body["thinking"] == {"type": "disabled"}
    else:
        assert body["max_completion_tokens"] == 16000
        assert "max_tokens" not in body and "thinking" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["deepseek", "openai"])
@pytest.mark.parametrize("stream", [True, False])
async def test_async_sdk_parameters_and_text_chunks(provider, stream):
    requests = []

    def handler(request):
        requests.append(request)
        return sdk_response(stream)

    client = openai.AsyncOpenAI(
        api_key=KEY, base_url=BASE, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    tracer = {}
    with (
        patch.object(utils, "get_openai_async_client", return_value=client),
        patch.dict(utils.openai_async_clients, {}, clear=True),
        patch.object(utils, "is_non_streaming_model", return_value=not stream),
    ):
        chunks = [
            c
            async for c in gpt.converse_openai(
                MESSAGES, "custom", KEY, BASE, provider_type=provider, deepthought=True, tracer=tracer
            )
        ]
    await client.close()
    assert all(isinstance(c, ResponseWithThought) for c in chunks)
    assert "".join(c.text for c in chunks) == "你好，我是 DeepSeek。"
    assert len(chunks) == (2 if stream else 1)
    assert tracer["usage"]
    assert len(requests) == 1
    assert str(requests[0].url) == BASE + "/chat/completions"
    body = json.loads(requests[0].content)
    assert body["model"] == "custom"
    if provider == "deepseek":
        assert body["max_tokens"] == 16000
        assert "max_completion_tokens" not in body
        assert body["thinking"] == {"type": "disabled"}
    else:
        assert body["max_completion_tokens"] == 16000
        assert "max_tokens" not in body and "thinking" not in body


@pytest.mark.parametrize(
    "provider,target", [("deepseek", "openai"), ("openai", "openai"), ("anthropic", "anthropic"), ("google", "gemini")]
)
def test_internal_dispatch(provider, target):
    model = ChatModel(name="custom", model_type=provider, ai_model_api=AiModelApi(api_key=KEY, api_base_url=BASE))
    with patch.object(helpers, f"{target}_send_message_to_model", return_value=ResponseWithThought(text="ok")) as send:
        result = helpers.send_message_to_model(model, MESSAGES, "text", None, None, False, {})
    assert result.text == "ok"
    assert send.call_args.kwargs["model"] == "custom"
    assert send.call_args.kwargs["api_key"] == KEY
    assert send.call_args.kwargs["api_base_url"] == BASE
    if provider == "deepseek":
        assert send.call_args.kwargs["provider_type"] == "deepseek"


@pytest.mark.parametrize(
    "provider,target", [("deepseek", "openai"), ("openai", "openai"), ("anthropic", "anthropic"), ("google", "gemini")]
)
def test_sync_wrapper_dispatch(provider, target):
    model = ChatModel(name="custom", model_type=provider, ai_model_api=AiModelApi(api_key=KEY, api_base_url=BASE))
    with (
        patch.object(ConversationAdapters, "get_default_chat_model", return_value=model),
        patch.object(ConversationAdapters, "get_max_context_size", return_value=60000),
        patch.object(helpers, "generate_chatml_messages_with_context", return_value=MESSAGES),
        patch.object(helpers, f"{target}_send_message_to_model", return_value=ResponseWithThought(text="ok")) as send,
    ):
        assert helpers.send_message_to_model_wrapper_sync("你好").text == "ok"
    assert send.call_args.kwargs["api_key"] == KEY
    assert send.call_args.kwargs["api_base_url"] == BASE
    if provider == "deepseek":
        assert send.call_args.kwargs["provider_type"] == "deepseek"


def test_dynamic_model_list(deepseek_env):
    from khoj.routers.api_model import get_chat_model_options

    register()
    response = get_chat_model_options(None)
    options = json.loads(response.body)
    assert {option["name"] for option in options} == {"deepseek-flash", "deepseek-v4-pro"}


@pytest.mark.parametrize("configured", [True, False])
def test_first_deployment_initialization(monkeypatch, configured):
    from khoj.database.models import SpeechToTextModelOptions, TextToImageModelConfig
    from khoj.utils.initialization import initialization

    for name in [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
        "KHOJ_DEFAULT_CHAT_MODEL",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KHOJ_ADMIN_EMAIL", "deepseek-test@example.test")
    monkeypatch.setenv("KHOJ_ADMIN_PASSWORD", "test-only-password")
    if configured:
        monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
    initialization(interactive=False)
    assert ChatModel.objects.filter(model_type="deepseek").count() == (2 if configured else 0)
    assert AiModelApi.objects.count() == (1 if configured else 0)
    if configured:
        assert AiModelApi.objects.get().api_base_url == "https://api.deepseek.com"
    assert not SpeechToTextModelOptions.objects.exists()
    assert not TextToImageModelConfig.objects.exists()


@pytest.mark.asyncio
async def test_final_chat_dispatch_and_chat_events():
    """Exercise /general through the real event generator and real SDK parser."""
    import asyncio
    from starlette.datastructures import Headers
    from khoj.routers import api_chat
    from khoj.processor.conversation.utils import ChatEvent
    from khoj.utils.rawconfig import ChatRequestBody

    requests = []

    def handler(request):
        requests.append(request)
        return sdk_response(True)

    client = openai.AsyncOpenAI(
        api_key=KEY, base_url=BASE, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    model = ChatModel(
        name="deepseek-flash",
        model_type="deepseek",
        max_prompt_size=60000,
        ai_model_api=AiModelApi(api_key=KEY, api_base_url=BASE),
    )
    default_agent = SimpleNamespace(id=1, slug="default")
    conversation = SimpleNamespace(id=1, agent=default_agent, messages=[], pop_message=AsyncMock(return_value=None))
    user_scope = SimpleNamespace(object=SimpleNamespace(id=1), client_app=None)
    save = AsyncMock()
    with (
        patch.object(ConversationAdapters, "aget_conversation_by_user", AsyncMock(return_value=conversation)),
        patch.object(ConversationAdapters, "aget_user_chat_model", AsyncMock(return_value=model)),
        patch.object(ConversationAdapters, "aget_chat_model", AsyncMock(return_value=model)),
        patch.object(ConversationAdapters, "aget_max_context_size", AsyncMock(return_value=60000)),
        patch.object(ConversationAdapters, "ais_memory_enabled", AsyncMock(return_value=False)),
        patch.object(AgentAdapters, "aget_default_agent", AsyncMock(return_value=default_agent)),
        patch.object(AgentAdapters, "aget_conversation_agent_by_id", AsyncMock(return_value=None)),
        patch.object(helpers, "build_conversation_context", return_value=MESSAGES),
        patch.object(api_chat, "aget_user_name", AsyncMock(return_value=None)),
        patch.object(api_chat, "has_required_scope", return_value=True),
        patch.object(api_chat, "update_telemetry_state"),
        patch.object(api_chat, "save_to_conversation_log", save),
        patch.object(api_chat.conversation_command_rate_limiter, "update_and_check_if_valid", AsyncMock()),
        patch.object(utils, "get_openai_async_client", return_value=client),
        patch.dict(utils.openai_async_clients, {}, clear=True),
    ):
        events = [
            event
            async for event in api_chat.event_generator(
                ChatRequestBody(q="/general 你好，请介绍一下自己", stream=True),
                user_scope,
                SimpleNamespace(client="test"),
                Headers(),
                SimpleNamespace(),
            )
        ]
        await asyncio.sleep(0)  # Let the existing background persistence task run.
    await client.close()
    assert "你好" in events and "，我是 DeepSeek。" in events
    structured = [json.loads(e) for e in events if e.startswith("{")]
    types = [e["type"] for e in structured]
    assert ChatEvent.START_LLM_RESPONSE.value in types
    assert types[-1] == ChatEvent.END_RESPONSE.value
    assert types.index(ChatEvent.START_LLM_RESPONSE.value) < types.index(ChatEvent.END_LLM_RESPONSE.value)
    assert events[-1] == ChatEvent.END_EVENT.value
    assert save.await_args.kwargs["chat_response"] == "你好，我是 DeepSeek。"
    body = json.loads(requests[0].content)
    assert body["model"] == "deepseek-flash"
    assert body["max_tokens"] == 16000 and "max_completion_tokens" not in body
    assert body["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize("provider", ["deepseek", "openai"])
def test_explicit_provider_controls_responses_routing(provider):
    with (
        patch.object(gpt, "supports_responses_api", return_value=True),
        patch.object(gpt, "completion_with_backoff", return_value=ResponseWithThought(text="chat")) as chat,
        patch.object(
            gpt, "responses_completion_with_backoff", return_value=ResponseWithThought(text="responses")
        ) as responses,
    ):
        result = gpt.openai_send_message_to_model(MESSAGES, KEY, "custom", api_base_url=BASE, provider_type=provider)
    assert result.text == ("chat" if provider == "deepseek" else "responses")
    assert chat.called == (provider == "deepseek")
    assert responses.called == (provider == "openai")


def test_registration_preserves_server_user_and_agent_selections(deepseek_env):
    from khoj.database.models import Agent, ServerChatSettings, UserConversationConfig
    from tests.helpers import ChatModelFactory, UserConversationProcessorConfigFactory

    original = ChatModelFactory()
    server = ServerChatSettings.objects.create(chat_default=original, chat_advanced=original)
    user_config = UserConversationProcessorConfigFactory(setting=original)
    agent = Agent.objects.create(name="Custom Agent", chat_model=original)
    snapshots = [list(cls.objects.values()) for cls in [ServerChatSettings, UserConversationConfig, Agent]]
    register()
    register()
    assert snapshots == [list(cls.objects.values()) for cls in [ServerChatSettings, UserConversationConfig, Agent]]
    server.refresh_from_db()
    user_config.refresh_from_db()
    agent.refresh_from_db()
    assert server.chat_default_id == user_config.setting_id == agent.chat_model_id == original.pk


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
@pytest.mark.parametrize("premium", [True, False])
async def test_model_selection_retains_subscription_rules(premium):
    from starlette.authentication import AuthCredentials
    from starlette.requests import Request
    from khoj.database.models import KhojUser, UserConversationConfig
    from khoj.routers.api_model import update_chat_model

    user = await KhojUser.objects.acreate(username="deepseek-select", email="select@example.test")
    api = await AiModelApi.objects.acreate(name="DeepSeek", api_key=KEY, api_base_url=BASE)
    model = await ChatModel.objects.acreate(
        name="deepseek-flash", model_type="deepseek", ai_model_api=api, price_tier=PriceTier.STANDARD
    )
    request = Request(
        {
            "type": "http",
            "auth": AuthCredentials(["authenticated"] + (["premium"] if premium else [])),
            "user": SimpleNamespace(object=user),
        }
    )
    with patch("khoj.routers.api_model.update_telemetry_state"):
        response = await update_chat_model(request, str(model.pk))
    if premium:
        assert response == {"status": "ok"}
        assert (await UserConversationConfig.objects.aget(user=user)).setting_id == model.pk
    else:
        assert response.status_code == 403
        assert not await UserConversationConfig.objects.filter(user=user).aexists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,target", [("deepseek", "openai"), ("openai", "openai"), ("anthropic", "anthropic"), ("google", "gemini")]
)
async def test_final_dispatch_preserves_other_providers(provider, target):
    model = ChatModel(name="custom", model_type=provider, ai_model_api=AiModelApi(api_key=KEY, api_base_url=BASE))

    async def output(**kwargs):
        yield ResponseWithThought(text="ok")

    with (
        patch.object(ConversationAdapters, "aget_valid_chat_model", AsyncMock(return_value=model)),
        patch.object(ConversationAdapters, "aget_max_context_size", AsyncMock(return_value=60000)),
        patch.object(helpers, "build_conversation_context", return_value=MESSAGES),
        patch.object(helpers, f"converse_{target}", side_effect=lambda *args, **kwargs: output(**kwargs)) as send,
    ):
        stream, metadata = await helpers.agenerate_chat_response("你好", [], SimpleNamespace(agent=None))
        assert [chunk.text async for chunk in stream] == ["ok"]
    assert send.call_args.kwargs["model"] == "custom"
    assert send.call_args.kwargs["api_key"] == KEY
    assert send.call_args.kwargs["api_base_url"] == BASE
    if provider == "deepseek":
        assert send.call_args.kwargs["provider_type"] == "deepseek"
    else:
        assert "provider_type" not in send.call_args.kwargs or provider == "openai"
    assert metadata["chat_model"] == "custom"


def test_default_provider_keeps_responses_api():
    with patch.object(gpt, "responses_completion_with_backoff", return_value=ResponseWithThought(text="ok")) as send:
        assert gpt.openai_send_message_to_model(MESSAGES, KEY, "gpt-4o-mini").text == "ok"
    assert "provider_type" not in send.call_args.kwargs


def test_prompt_budget_and_tokenizer_fallback():
    import tiktoken
    from khoj.processor.conversation.utils import model_to_prompt_size
    from khoj.utils.helpers import get_encoder

    for name in ["deepseek-flash", "deepseek-v4-pro"]:
        assert model_to_prompt_size[name] == 60000
        assert get_encoder(name) is tiktoken.encoding_for_model("gpt-4o")


def test_deepseek_migration_applied_by_test_database():
    from django.db.migrations.recorder import MigrationRecorder

    assert MigrationRecorder.Migration.objects.filter(app="database", name="0100_add_deepseek_chat_model_type").exists()


def test_invalid_deepseek_startup_fails_once_and_rolls_back_other_providers(monkeypatch):
    from khoj.database.models import SpeechToTextModelOptions, TextToImageModelConfig
    from khoj.utils import initialization as setup

    for name in ["OPENAI_BASE_URL", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "KHOJ_DEFAULT_CHAT_MODEL"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KHOJ_ADMIN_EMAIL", "invalid-deepseek@example.test")
    monkeypatch.setenv("KHOJ_ADMIN_PASSWORD", "test-only-password")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-openai-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "invalid-url")
    # A permanent config error must escape; stop the legacy loop if it tries again.
    with patch.object(setup.logger, "error", side_effect=AssertionError("Permanent configuration error was retried")):
        with pytest.raises(ValueError, match="DEEPSEEK_BASE_URL"):
            setup.initialization(interactive=False)
    assert not AiModelApi.objects.exists()
    assert not ChatModel.objects.exists()
    assert not SpeechToTextModelOptions.objects.exists()
    assert not TextToImageModelConfig.objects.exists()

"""Offline regressions for JSON selection over Chat Completions.

These tests mock transport and database boundaries, never a paid model API.
"""

import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest
from langchain_core.messages.chat import ChatMessage
from pydantic import BaseModel, Field

from khoj.database.adapters import AgentAdapters, ConversationAdapters, EntryAdapters
from khoj.database.models import AiModelApi, ChatModel
from khoj.processor.conversation.openai import gpt, utils
from khoj.processor.conversation.utils import ResponseWithThought, StructuredOutputSupport
from khoj.routers import api_chat, helpers
from khoj.utils.helpers import ConversationCommand, ToolDefinition
from khoj.utils.rawconfig import ChatRequestBody

KEY = "synthetic-structured-output-key"
BASE = "https://api.deepseek.com"
VALID_SELECTION = '{"source":["general"],"output":"text"}'


class Selection(BaseModel):
    source: list[str] = Field(min_length=1)
    output: str


def streaming_response(text):
    chunk = {
        "id": "offline",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "offline-model",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    return httpx.Response(
        200,
        text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        headers={"content-type": "text/event-stream"},
    )


@contextmanager
def mock_sdk(base=BASE, text=VALID_SELECTION, status=200, max_retries=2):
    requests = []

    def handler(request):
        requests.append(request)
        if status != 200:
            return httpx.Response(
                status,
                json={"error": {"message": "synthetic API error", "type": "invalid_request_error"}},
            )
        return streaming_response(text)

    with openai.OpenAI(
        api_key=KEY,
        base_url=base,
        max_retries=max_retries,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ) as client:
        with (
            patch.object(utils, "get_openai_client", return_value=client),
            patch.dict(utils.openai_clients, {}, clear=True),
            patch.object(utils, "is_promptrace_enabled", return_value=False),
        ):
            yield requests


@pytest.mark.parametrize("model", ["deepseek-flash", "deepseek-v4-pro", "deepseek-chat", "renamed-model"])
@pytest.mark.parametrize("base", [BASE, BASE + "/v1", "https://proxy.example.test/v1"])
@pytest.mark.parametrize(
    "response_type,schema,expected_format",
    [
        ("json_object", Selection, "json_object"),
        (None, Selection, "json_object"),
        ("json_object", None, "json_object"),
        ("text", None, None),
    ],
)
def test_deepseek_wire_format(model, base, response_type, schema, expected_format):
    with mock_sdk(base) as requests:
        result = gpt.openai_send_message_to_model(
            [ChatMessage(role="user", content="Return JSON selection")],
            KEY,
            model,
            response_type=response_type,
            response_schema=schema,
            api_base_url=base,
            provider_type="deepseek",
        )
    assert result.text == VALID_SELECTION
    assert len(requests) == 1
    assert str(requests[0].url) == base + "/chat/completions"
    body = json.loads(requests[0].content)
    assert body["model"] == model
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["max_tokens"] == 16000
    assert "max_completion_tokens" not in body
    assert body["thinking"] == {"type": "disabled"}
    assert "tools" not in body
    if expected_format:
        assert body["response_format"] == {"type": expected_format}
    else:
        assert "response_format" not in body


@pytest.mark.parametrize("base", [BASE, BASE + "/v1", "https://API.DeepSeek.COM/v1"])
def test_legacy_official_host_uses_object(base):
    assert utils.get_structured_output_support("alias", base) == StructuredOutputSupport.OBJECT
    with mock_sdk(base) as requests:
        gpt.openai_send_message_to_model(
            [ChatMessage(role="user", content="Return JSON")],
            KEY,
            "alias",
            response_type="json_object",
            response_schema=Selection,
            api_base_url=base,
        )
    assert json.loads(requests[0].content)["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("provider", [None, "openai", "deepseek"])
def test_old_reasoner_exception_is_preserved(provider):
    assert (
        utils.get_structured_output_support("deepseek-reasoner", BASE, provider_type=provider)
        == StructuredOutputSupport.NONE
    )


def test_old_reasoner_wire_does_not_enable_structured_output():
    with mock_sdk() as requests:
        result = gpt.openai_send_message_to_model(
            [ChatMessage(role="user", content="Return JSON")],
            KEY,
            "deepseek-reasoner",
            response_type="json_object",
            response_schema=Selection,
            api_base_url=BASE,
            provider_type="deepseek",
        )
    assert result.text == VALID_SELECTION
    assert "response_format" not in json.loads(requests[0].content)


@pytest.mark.parametrize(
    "base,expected",
    [
        (None, StructuredOutputSupport.TOOL),
        ("https://api.openai.com/v1", StructuredOutputSupport.TOOL),
        ("https://api.deepinfra.com/v1", StructuredOutputSupport.OBJECT),
        ("https://deployment.ai.azure.com/v1", StructuredOutputSupport.OBJECT),
        ("https://api.deepseek.com.evil.test/v1", StructuredOutputSupport.TOOL),
        ("https://api.deepseek.com@evil.test/v1", StructuredOutputSupport.TOOL),
        ("https://example.test/api.deepseek.com", StructuredOutputSupport.TOOL),
    ],
)
def test_other_hosts_keep_existing_capabilities(base, expected):
    assert utils.get_structured_output_support("alias", base) == expected


@pytest.mark.parametrize("base", ["https://deployment.ai.azure.com/v1", "https://api.deepinfra.com/v1"])
def test_existing_object_hosts_keep_json_output(base):
    with mock_sdk(base) as requests:
        gpt.openai_send_message_to_model(
            [ChatMessage(role="user", content="Return JSON")],
            KEY,
            "custom",
            response_type="json_object",
            response_schema=Selection,
            api_base_url=base,
        )
    assert json.loads(requests[0].content)["response_format"] == {"type": "json_object"}


def test_deepseek_chat_capability_does_not_claim_responses_support():
    assert (
        utils.get_structured_output_support("alias", BASE, provider_type="deepseek", use_responses_api=True)
        == StructuredOutputSupport.TOOL
    )


@pytest.mark.parametrize("base", ["https://proxy.example.test/v1", "https://api.deepinfra.com/v1"])
def test_schema_only_adds_instructions_and_validates_locally(base):
    messages = [ChatMessage(role="system", content="Synthetic system"), ChatMessage(role="user", content="Choose")]
    before = [m.model_dump() for m in messages]
    with mock_sdk(base) as requests:
        gpt.openai_send_message_to_model(
            messages,
            KEY,
            "alias",
            response_type=None,
            response_schema=Selection,
            api_base_url=base,
            provider_type="deepseek",
        )
    body = json.loads(requests[0].content)
    assert "JSON" in body["messages"][0]["content"]
    assert '"source"' in body["messages"][0]["content"]
    assert '"output"' in body["messages"][0]["content"]
    assert [m.model_dump() for m in messages] == before
    with mock_sdk(base, text='{"source":"general","output":"text"}') as invalid_requests:
        with pytest.raises(ValueError, match="does not match the requested JSON schema"):
            gpt.openai_send_message_to_model(
                messages,
                KEY,
                "alias",
                response_type=None,
                response_schema=Selection,
                api_base_url=base,
                provider_type="deepseek",
            )
    assert len(invalid_requests) == 1


def test_schema_only_validation_error_does_not_disclose_output():
    with mock_sdk(text='{"source":"private-model-output","output":"text"}'):
        with pytest.raises(ValueError) as error:
            gpt.openai_send_message_to_model(
                [ChatMessage(role="user", content="Choose")],
                KEY,
                "alias",
                response_type=None,
                response_schema=Selection,
                api_base_url=BASE,
                provider_type="deepseek",
            )
    assert "private-model-output" not in str(error.value)


@pytest.mark.parametrize("base", [BASE, "https://proxy.example.test/v1"])
def test_deepseek_tools_are_explicitly_unsupported(base):
    tool = ToolDefinition(name="lookup", description="Synthetic lookup", schema=Selection)
    with mock_sdk(base) as requests:
        with pytest.raises(NotImplementedError, match="DeepSeek.*tool"):
            gpt.openai_send_message_to_model(
                [ChatMessage(role="user", content="Choose")],
                KEY,
                "alias",
                tools=[tool],
                response_type="json_object",
                response_schema=Selection,
                api_base_url=base,
                provider_type="deepseek",
            )
    assert not requests


@pytest.mark.parametrize("base", [None, "https://proxy.example.test/v1"])
@pytest.mark.parametrize("mode", ["schema", "object", "tools", "text"])
def test_openai_structured_output_and_tools_preserved(base, mode):
    requests = []

    def reject(request):
        requests.append(request)
        return httpx.Response(400, json={"error": {"message": "synthetic rejection"}})

    with openai.OpenAI(
        api_key=KEY, base_url=base, http_client=httpx.Client(transport=httpx.MockTransport(reject))
    ) as client:
        with (
            patch.object(utils, "get_openai_client", return_value=client),
            patch.dict(utils.openai_clients, {}, clear=True),
        ):
            with pytest.raises(openai.BadRequestError):
                gpt.openai_send_message_to_model(
                    [ChatMessage(role="user", content="Return JSON")],
                    KEY,
                    "gpt-4o-mini",
                    response_type="json_object" if mode != "text" else "text",
                    response_schema=Selection if mode in ("schema", "tools") else None,
                    tools=[ToolDefinition(name="lookup", description="Lookup", schema=Selection)]
                    if mode == "tools"
                    else None,
                    api_base_url=base,
                )
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    if base is None:
        assert requests[0].url.path == "/v1/responses"
        fmt = body.get("text", {}).get("format")
    else:
        assert requests[0].url.path == "/v1/chat/completions"
        fmt = body.get("response_format")
    if mode in ("schema", "object"):
        assert fmt["type"] == ("json_schema" if mode == "schema" else "json_object")
    else:
        assert fmt is None
    if mode == "tools":
        assert body["tools"][0]["type"] == "function"


@contextmanager
def planner_model():
    model = ChatModel(
        id=1,
        name="deepseek-flash",
        model_type="deepseek",
        max_prompt_size=60000,
        ai_model_api=AiModelApi(api_key=KEY, api_base_url=BASE),
    )
    with (
        patch.object(EntryAdapters, "auser_has_entries", AsyncMock(return_value=True)),
        patch.object(ConversationAdapters, "aget_default_chat_model", AsyncMock(return_value=model)),
        patch.object(ConversationAdapters, "aget_chat_model_slot", AsyncMock(return_value=None)),
        patch.object(ConversationAdapters, "aget_chat_models_with_fallbacks", AsyncMock(return_value=[])),
        patch.object(ConversationAdapters, "aget_max_context_size", AsyncMock(return_value=60000)),
        patch.object(
            helpers,
            "generate_chatml_messages_with_context",
            side_effect=lambda **kw: [ChatMessage(role="user", content=kw["user_message"])],
        ),
        patch.object(helpers, "is_web_search_enabled", return_value=False),
        patch.object(helpers, "is_operator_enabled", return_value=False),
        patch.object(helpers, "is_code_sandbox_enabled", return_value=False),
    ):
        yield model


def test_real_planner_to_sdk_uses_json_object():
    async def run():
        with planner_model(), mock_sdk() as requests:
            result = await helpers.aget_data_sources_and_output_format("你好，请介绍一下自己。", [], None)
        assert result == {"sources": [ConversationCommand.General], "output": ConversationCommand.Text}
        assert len(requests) == 1
        body = json.loads(requests[0].content)
        assert body["response_format"] == {"type": "json_object"}
        assert "JSON" in body["messages"][0]["content"]

    asyncio.run(run())


@pytest.mark.parametrize(
    "text",
    [
        "not-json-private-sentinel",
        '{"source":["notes"]}',
        '{"output":"text"}',
        '{"source":"general","output":"text"}',
        '{"source":[],"output":"text"}',
        '{"source":[1],"output":"text"}',
        '{"source":["general"],"output":3}',
        '{"source":["general"],"output":null}',
        '{"source":[" "],"output":"text"}',
        '{"source":["notes"],"output":""}',
        "",
        None,
    ],
)
def test_invalid_picktools_falls_back_without_private_logs(text, caplog):
    async def run():
        with (
            planner_model(),
            patch.object(
                helpers, "send_message_to_model_wrapper", AsyncMock(return_value=ResponseWithThought(text=text))
            ),
        ):
            result = await helpers.aget_data_sources_and_output_format("private-user-query", [], None)
        assert result == {"sources": [ConversationCommand.General], "output": ConversationCommand.Text}

    asyncio.run(run())
    assert "private-user-query" not in caplog.text
    assert "private-sentinel" not in caplog.text
    assert "UnboundLocalError" not in caplog.text
    if text != '{"source":[" "],"output":"text"}':
        assert "Invalid tool selection response" in caplog.text


@pytest.mark.parametrize(
    "text,expected_sources,expected_output",
    [
        (
            '{"source":["general","invented-tool"],"output":"text"}',
            [ConversationCommand.General],
            ConversationCommand.Text,
        ),
        ('{"source":["invented-tool"],"output":"image"}', [ConversationCommand.General], ConversationCommand.Text),
        ('{"source":["general"],"output":"invented-output"}', [ConversationCommand.General], ConversationCommand.Text),
        (
            '```json\n{"source":[" notes "],"output":" text "}\n```',
            [ConversationCommand.Notes],
            ConversationCommand.Text,
        ),
    ],
)
def test_picktools_filters_options(text, expected_sources, expected_output):
    async def run():
        with (
            planner_model(),
            patch.object(
                helpers, "send_message_to_model_wrapper", AsyncMock(return_value=ResponseWithThought(text=text))
            ),
        ):
            result = await helpers.aget_data_sources_and_output_format("synthetic query", [], None)
        assert result == {"sources": expected_sources, "output": expected_output}
        assert all(isinstance(source, ConversationCommand) for source in result["sources"])

    asyncio.run(run())


def test_picktools_preserves_agent_allowlist():
    async def run():
        agent = SimpleNamespace(input_tools=["general"], output_modes=["text"], personality=None)
        with (
            planner_model(),
            patch.object(AgentAdapters, "get_agent_chat_model", return_value=None),
            patch.object(
                helpers,
                "send_message_to_model_wrapper",
                AsyncMock(return_value=ResponseWithThought(text='{"source":["notes","general"],"output":"image"}')),
            ),
        ):
            result = await helpers.aget_data_sources_and_output_format("synthetic query", [], None, agent=agent)
        assert result == {"sources": [ConversationCommand.General], "output": ConversationCommand.Text}

    asyncio.run(run())


@pytest.mark.parametrize("status,exception", [(400, openai.BadRequestError), (401, openai.AuthenticationError)])
def test_api_errors_propagate_once_from_planner(status, exception):
    async def run():
        with planner_model(), mock_sdk(status=status) as requests:
            with pytest.raises(exception):
                await helpers.aget_data_sources_and_output_format("synthetic query", [], None)
        assert len(requests) == 1

    asyncio.run(run())


@pytest.mark.parametrize("status,exception", [(429, openai.RateLimitError), (500, openai.InternalServerError)])
def test_transient_errors_keep_bounded_completion_retries(status, exception):
    with (
        mock_sdk(status=status, max_retries=0) as requests,
        patch.object(utils.completion_with_backoff.retry, "sleep", lambda _: None),
    ):
        with pytest.raises(exception):
            gpt.openai_send_message_to_model(
                [ChatMessage(role="user", content="Return JSON")],
                KEY,
                "deepseek-flash",
                response_type="json_object",
                response_schema=Selection,
                api_base_url=BASE,
                provider_type="deepseek",
            )
    assert len(requests) == 2


@pytest.mark.parametrize(
    "query,expected_requests",
    [
        ("你好，请介绍一下自己。", 1),
        ("/default 你好", 1),
        ("/general 你好", 0),
    ],
)
def test_event_generator_reaches_answer_after_real_selection(query, expected_requests):
    from starlette.datastructures import Headers

    async def run():
        default_agent = SimpleNamespace(id=1, slug="default")
        conversation = SimpleNamespace(id=1, agent=default_agent, messages=[], pop_message=AsyncMock(return_value=None))

        async def answer():
            yield ResponseWithThought(text="Mock final answer")

        save = AsyncMock()
        with (
            planner_model(),
            mock_sdk() as requests,
            patch.object(ConversationAdapters, "aget_conversation_by_user", AsyncMock(return_value=conversation)),
            patch.object(ConversationAdapters, "ais_memory_enabled", AsyncMock(return_value=False)),
            patch.object(AgentAdapters, "aget_default_agent", AsyncMock(return_value=default_agent)),
            patch.object(api_chat, "is_ready_to_chat", AsyncMock()),
            patch.object(api_chat, "aget_user_name", AsyncMock(return_value=None)),
            patch.object(api_chat, "has_required_scope", return_value=True),
            patch.object(api_chat, "update_telemetry_state"),
            patch.object(api_chat, "agenerate_chat_response", AsyncMock(return_value=(answer(), {}))),
            patch.object(api_chat, "save_to_conversation_log", save),
            patch.object(api_chat.conversation_command_rate_limiter, "update_and_check_if_valid", AsyncMock()),
        ):
            events = [
                event
                async for event in api_chat.event_generator(
                    ChatRequestBody(q=query, stream=True),
                    SimpleNamespace(object=SimpleNamespace(id=1), client_app=None),
                    SimpleNamespace(client="test"),
                    Headers(),
                    SimpleNamespace(),
                )
            ]
            await asyncio.sleep(0)
        assert "Mock final answer" in events
        assert len(requests) == expected_requests
        assert save.await_args.kwargs["chat_response"] == "Mock final answer"
        if requests:
            assert json.loads(requests[0].content)["response_format"] == {"type": "json_object"}

    asyncio.run(run())

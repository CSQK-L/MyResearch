import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_core.messages.chat import ChatMessage
from pydantic import ValidationError

from khoj.processor.conversation.openai.utils import (
    chat_completion_with_backoff,
    clean_response_schema,
    completion_with_backoff,
    get_structured_output_support,
    is_cerebras_api,
    is_deepseek_api,
    responses_chat_completion_with_backoff,
    responses_completion_with_backoff,
    supports_responses_api,
    to_openai_tools,
)
from khoj.processor.conversation.utils import (
    ResponseWithThought,
    StructuredOutputSupport,
    clean_json,
    messages_to_print,
)
from khoj.utils.helpers import ToolDefinition

logger = logging.getLogger(__name__)


def openai_send_message_to_model(
    messages,
    api_key,
    model: str,
    response_type="text",
    response_schema=None,
    tools: list[ToolDefinition] = None,
    deepthought=False,
    api_base_url: str | None = None,
    tracer: dict = {},
    provider_type: str = "openai",
):
    """
    Send message to model
    """

    model_kwargs: Dict[str, Any] = {}
    use_responses_api = provider_type != "deepseek" and supports_responses_api(model, api_base_url)
    json_support = get_structured_output_support(
        model, api_base_url, provider_type=provider_type, use_responses_api=use_responses_api
    )
    if tools and not use_responses_api and (provider_type == "deepseek" or is_deepseek_api(api_base_url)):
        raise NotImplementedError("DeepSeek tool calling is not supported by this text-chat integration.")
    # Schema-only calls need explicit instructions and local validation when the
    # endpoint can enforce JSON syntax but cannot enforce a schema.
    validate_object_response = bool(
        response_schema and json_support == StructuredOutputSupport.OBJECT and response_type != "json_object"
    )
    strict = not is_cerebras_api(api_base_url)
    if tools and json_support == StructuredOutputSupport.TOOL:
        model_kwargs["tools"] = to_openai_tools(tools, model=model, api_base_url=api_base_url)
    elif response_schema and json_support >= StructuredOutputSupport.SCHEMA:
        # Drop unsupported fields from schema passed to OpenAI APi
        cleaned_response_schema = clean_response_schema(response_schema)
        if use_responses_api:
            model_kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "strict": strict,
                    "name": response_schema.__name__,
                    "schema": cleaned_response_schema,
                }
            }
        else:
            model_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "schema": cleaned_response_schema,
                    "name": response_schema.__name__,
                    "strict": strict,
                },
            }
    elif (response_schema or response_type == "json_object") and json_support >= StructuredOutputSupport.OBJECT:
        if use_responses_api:
            model_kwargs["text"] = {"format": {"type": "json_object"}}
        else:
            model_kwargs["response_format"] = {"type": "json_object"}

    if validate_object_response:
        instructions = "Respond with a JSON object matching this JSON schema:\n" + json.dumps(
            response_schema.model_json_schema()
        )
        if messages and messages[0].role == "system":
            content = messages[0].content
            content = (
                f"{content}\n\n{instructions}"
                if isinstance(content, str)
                else [*content, {"type": "text", "text": instructions}]
            )
            messages = [messages[0].model_copy(update={"content": content}), *messages[1:]]
        else:
            messages = [ChatMessage(role="system", content=instructions), *messages]

    # Get Response from GPT
    if use_responses_api:
        return responses_completion_with_backoff(
            messages=messages,
            model_name=model,
            openai_api_key=api_key,
            api_base_url=api_base_url,
            deepthought=deepthought,
            model_kwargs=model_kwargs,
            tracer=tracer,
        )
    else:
        response = completion_with_backoff(
            messages=messages,
            model_name=model,
            openai_api_key=api_key,
            api_base_url=api_base_url,
            deepthought=deepthought,
            model_kwargs=model_kwargs,
            tracer=tracer,
            provider_type=provider_type,
        )
        if validate_object_response:
            text = clean_json(response.text) if isinstance(response.text, str) else response.text
            try:
                response_schema.model_validate_json(text, strict=True)
            except ValidationError:
                # Upstream fallback logging must not include model output from
                # Pydantic's input_value diagnostics.
                raise ValueError("Model response does not match the requested JSON schema") from None
        return response


async def converse_openai(
    # Query
    messages: List[ChatMessage],
    # Model
    model: str = "gpt-4.1-mini",
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    temperature: float = 0.6,
    deepthought: Optional[bool] = False,
    tracer: dict = {},
    provider_type: str = "openai",
) -> AsyncGenerator[ResponseWithThought, None]:
    """
    Converse with user using OpenAI's ChatGPT
    """
    logger.debug(f"Conversation Context for GPT: {messages_to_print(messages)}")

    # Get Response from GPT
    if provider_type != "deepseek" and supports_responses_api(model, api_base_url):
        async for chunk in responses_chat_completion_with_backoff(
            messages=messages,
            model_name=model,
            temperature=temperature,
            openai_api_key=api_key,
            api_base_url=api_base_url,
            deepthought=deepthought,
            tracer=tracer,
        ):
            yield chunk
    else:
        # For non-OpenAI APIs, use the chat completion method
        async for chunk in chat_completion_with_backoff(
            messages=messages,
            model_name=model,
            temperature=temperature,
            openai_api_key=api_key,
            api_base_url=api_base_url,
            deepthought=deepthought,
            tracer=tracer,
            provider_type=provider_type,
        ):
            yield chunk

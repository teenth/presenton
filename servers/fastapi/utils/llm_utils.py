import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Optional

import dirtyjson
from fastapi import HTTPException
from llmai.shared import (
    LLMTool,
    Message,
    ResponseFormat,
    UserMessage,
    normalize_content_parts,
)

from enums.llm_provider import LLMProvider
from utils.llm_config import get_extra_body
from utils.get_env import (
    get_azure_openai_base_url_env,
    get_cerebras_base_url_env,
    get_custom_llm_url_env,
    get_fireworks_base_url_env,
    get_litellm_base_url_env,
    get_llm_provider_env,
    get_lmstudio_base_url_env,
    get_openrouter_base_url_env,
    get_ollama_url_env,
    get_vertex_base_url_env,
    get_together_base_url_env,
)
from utils.schema_utils import get_schema_validation_errors


LOGGER = logging.getLogger(__name__)


def llm_debug_logs_enabled() -> bool:
    return (os.getenv("LLM_DEBUG_LOGS") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _safe_json_preview(value: Any, max_chars: int = 1200) -> str:
    try:
        preview = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        preview = repr(value)
    if len(preview) > max_chars:
        return preview[:max_chars] + "...(truncated)"
    return preview


def _normalize_endpoint_url(base_url: Optional[str]) -> Optional[str]:
    if not base_url:
        return None

    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _infer_client_base_url(client: Any) -> Optional[str]:
    candidate_attrs = ("base_url", "_base_url", "api_base", "_api_base", "api_url")
    for attr in candidate_attrs:
        base_url = getattr(client, attr, None)
        if isinstance(base_url, str) and base_url.strip():
            return base_url

    nested_client = getattr(client, "client", None)
    if nested_client and nested_client is not client:
        for attr in candidate_attrs:
            nested_value = getattr(nested_client, attr, None)
            if isinstance(nested_value, str) and nested_value.strip():
                return nested_value

    return None


def _infer_base_url_from_provider() -> Optional[str]:
    provider_name = (get_llm_provider_env() or "").strip().lower()
    if not provider_name:
        return None

    try:
        provider = LLMProvider(provider_name)
    except Exception:
        return None

    if provider == LLMProvider.CUSTOM:
        return get_custom_llm_url_env()
    if provider == LLMProvider.OPENROUTER:
        return get_openrouter_base_url_env()
    if provider == LLMProvider.FIREWORKS:
        return get_fireworks_base_url_env()
    if provider == LLMProvider.TOGETHER:
        return get_together_base_url_env()
    if provider == LLMProvider.CEREBRAS:
        return get_cerebras_base_url_env()
    if provider == LLMProvider.LITELLM:
        return get_litellm_base_url_env()
    if provider == LLMProvider.LMSTUDIO:
        return get_lmstudio_base_url_env()
    if provider == LLMProvider.VERTEX:
        return get_vertex_base_url_env()
    if provider == LLMProvider.AZURE:
        return get_azure_openai_base_url_env()
    if provider == LLMProvider.OLLAMA:
        ollama_url = (get_ollama_url_env() or "").strip()
        if ollama_url:
            return f"{ollama_url.rstrip('/')}/v1"
    return None


def _describe_response_format(response_format: Any) -> Optional[dict[str, Any]]:
    if response_format is None:
        return None

    description: dict[str, Any] = {
        "class": response_format.__class__.__name__,
    }
    for attr in ("type", "name", "strict"):
        value = getattr(response_format, attr, None)
        if value is not None:
            description[attr] = value

    json_schema = getattr(response_format, "json_schema", None)
    if isinstance(json_schema, dict):
        description["json_schema_keys"] = sorted(json_schema.keys())
        properties = json_schema.get("properties")
        if isinstance(properties, dict):
            description["json_schema_properties"] = sorted(properties.keys())

    return description


def _describe_generate_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    tools = kwargs.get("tools") or []
    extra_body = kwargs.get("extra_body")

    description: dict[str, Any] = {
        "model": kwargs.get("model"),
        "stream": kwargs.get("stream"),
        "message_count": len(kwargs.get("messages") or []),
        "tool_classes": [tool.__class__.__name__ for tool in tools],
        "response_format": _describe_response_format(kwargs.get("response_format")),
    }
    if "max_tokens" in kwargs:
        description["max_tokens"] = kwargs.get("max_tokens")
    if isinstance(extra_body, dict):
        description["extra_body_keys"] = sorted(extra_body.keys())
    elif extra_body is not None:
        description["extra_body_type"] = extra_body.__class__.__name__

    return description


def get_generate_kwargs(
    model: str,
    messages: Sequence[Message],
    max_tokens: Optional[int] = None,
    tools: Optional[list[LLMTool]] = None,
    response_format: Optional[ResponseFormat] = None,
    stream: bool = False,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "stream": stream,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if tools:
        kwargs["tools"] = tools
    if response_format is not None:
        kwargs["response_format"] = response_format

    extra_body = get_extra_body()
    if extra_body:
        kwargs["extra_body"] = extra_body

    return kwargs


def structured_validation_feedback_user_message(
    content: dict,
    validation_errors: list[str],
) -> UserMessage:
    max_error_count = 10
    max_json_chars = 6000

    formatted_errors = validation_errors[:max_error_count]
    if len(validation_errors) > max_error_count:
        formatted_errors.append(
            f"...and {len(validation_errors) - max_error_count} more validation errors."
        )

    previous_response = json.dumps(
        content,
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    if len(previous_response) > max_json_chars:
        previous_response = previous_response[:max_json_chars] + "\n... (truncated)"

    return UserMessage(
        content=(
            "The previous JSON response did not match the required response schema.\n\n"
            "Validation errors:\n"
            + "\n".join(f"- {error}" for error in formatted_errors)
            + "\n\nPrevious invalid JSON:\n"
            + f"```json\n{previous_response}\n```\n\n"
            + "Return corrected JSON only. Make sure it fully matches the required schema."
        )
    )


async def generate_structured_with_schema_retries(
    client: Any,
    model: str,
    *,
    messages: Sequence[Message],
    response_format: ResponseFormat,
    json_schema: dict,
    strict: bool = False,
    validate_schema: bool = False,
    validate_schema_max_loop_count: int = 4,
) -> dict:
    """
    Parse retries (inner loop) plus optional JSON Schema validation feedback loops (outer loop),
    matching the overflow-mitigation behavior from structured generation with validate_schema.
    """
    max_validation_loops = max(1, validate_schema_max_loop_count)
    working_messages: list[Message] = list(messages)

    for validation_attempt in range(max_validation_loops):
        content: Optional[dict] = None
        for attempt in range(3):
            response = await asyncio.to_thread(
                client.generate,
                **get_generate_kwargs(
                    model=model,
                    messages=working_messages,
                    response_format=response_format,
                ),
            )
            content = extract_structured_content(response.content)
            if content is not None:
                break
            if attempt < 2:
                await asyncio.sleep(0.5 * (attempt + 1))

        if content is None:
            raise HTTPException(
                status_code=400,
                detail="LLM did not return any content",
            )

        if not validate_schema:
            return content

        validation_errors = get_schema_validation_errors(
            json_schema,
            content,
            strict=strict,
        )

        if not validation_errors:
            return content

        formatted_validation_errors = " | ".join(validation_errors)
        if validation_attempt == max_validation_loops - 1:
            LOGGER.warning(
                "Validation error after max fixes, returning last response: %s",
                formatted_validation_errors,
            )
            return content

        LOGGER.warning(
            "Validation error, attempting fix %s/%s: %s",
            validation_attempt + 1,
            max_validation_loops - 1,
            formatted_validation_errors,
        )
        working_messages.append(
            structured_validation_feedback_user_message(content, validation_errors)
        )

    raise HTTPException(status_code=400, detail="LLM did not return any content")


def extract_text(content: Any) -> Optional[str]:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
        joined = "".join(parts)
        return joined or None
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text
    return None


def extract_structured_content(content: Any) -> Optional[dict]:
    if content is None:
        return None
    if isinstance(content, dict):
        return content
    if hasattr(content, "model_dump"):
        dumped = content.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped

    raw_text = extract_text(content)
    if not raw_text:
        return None

    try:
        parsed = dirtyjson.loads(raw_text)
    except Exception:
        return None

    if isinstance(parsed, dict):
        return dict(parsed)
    return None


def serialize_structured_content(content: Any) -> Optional[str]:
    parsed = extract_structured_content(content)
    if parsed is not None:
        return json.dumps(parsed, ensure_ascii=False)

    raw_text = extract_text(content)
    if raw_text:
        return raw_text
    return None


def message_content_to_text(content: Sequence[Any] | str | None) -> Optional[str]:
    joined = "".join(
        part.text
        for part in normalize_content_parts(content)
        if isinstance(getattr(part, "text", None), str)
    )
    return joined or None


async def stream_generate_events(client: Any, **kwargs) -> AsyncGenerator[Any, None]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    sentinel = object()
    endpoint = _normalize_endpoint_url(_infer_client_base_url(client))
    if endpoint is None:
        try:
            endpoint = _normalize_endpoint_url(_infer_base_url_from_provider())
        except Exception:
            endpoint = None

    def worker():
        if llm_debug_logs_enabled():
            LOGGER.info(
                "[llm-debug] client.generate start: client=%s endpoint=%s kwargs=%s",
                client.__class__.__name__,
                endpoint,
                _safe_json_preview(_describe_generate_kwargs(kwargs)),
            )
        event_count = 0
        try:
            for event in client.generate(**kwargs):
                event_count += 1
                if llm_debug_logs_enabled() and event_count <= 3:
                    LOGGER.info(
                        "[llm-debug] client.generate event[%s]: class=%s type=%s",
                        event_count,
                        event.__class__.__name__,
                        getattr(event, "type", None),
                    )
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:
            if llm_debug_logs_enabled():
                LOGGER.exception(
                    "[llm-debug] client.generate failed: class=%s message=%s "
                    "events_seen=%s endpoint=%s request=%s",
                    exc.__class__.__name__,
                    str(exc),
                    event_count,
                    endpoint,
                    _safe_json_preview(_describe_generate_kwargs(kwargs)),
                )
            else:
                LOGGER.error(
                    "[llm-error] client.generate failed: class=%s endpoint=%s "
                    "events_seen=%s request=%s",
                    exc.__class__.__name__,
                    endpoint,
                    event_count,
                    _safe_json_preview(_describe_generate_kwargs(kwargs)),
                )
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            if llm_debug_logs_enabled():
                LOGGER.info(
                    "[llm-debug] client.generate finished: events_seen=%s",
                    event_count,
                )
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    worker_task = asyncio.create_task(asyncio.to_thread(worker))
    try:
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        await worker_task

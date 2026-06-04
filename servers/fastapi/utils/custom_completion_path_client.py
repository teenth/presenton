import json
import logging
from types import SimpleNamespace
from typing import Any, Iterator, Optional
from urllib.parse import urlsplit

import httpx

from utils.get_env import (
    get_custom_llm_api_key_env,
    get_custom_llm_completion_path_env,
    get_custom_llm_url_env,
)


LOGGER = logging.getLogger(__name__)


def build_custom_completion_endpoint(
    base_url: Optional[str],
    completion_path: Optional[str],
) -> Optional[str]:
    base = (base_url or "").strip().rstrip("/")
    path = (completion_path or "").strip()
    if not base or not path:
        return None
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return f"{base}/{path.lstrip('/')}"


def _message_role(message: Any) -> str:
    class_name = message.__class__.__name__.lower()
    if "system" in class_name:
        return "system"
    if "assistant" in class_name:
        return "assistant"
    if "tool" in class_name:
        return "tool"
    return "user"


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(part for part in parts if part)
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text
    return str(content)


def _messages_to_payload_messages(messages: list[Any]) -> list[dict[str, str]]:
    payload_messages: list[dict[str, str]] = []
    for message in messages:
        payload_messages.append(
            {
                "role": _message_role(message),
                "content": _content_to_text(getattr(message, "content", message)),
            }
        )
    return payload_messages


def _response_format_for_chat(response_format: Any) -> Optional[dict[str, Any]]:
    if response_format is None:
        return None
    json_schema = getattr(response_format, "json_schema", None)
    if isinstance(json_schema, dict):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": getattr(response_format, "name", None) or "response",
                "schema": json_schema,
                "strict": bool(getattr(response_format, "strict", False)),
            },
        }
    return {"type": "text"}


def _response_format_for_responses(response_format: Any) -> Optional[dict[str, Any]]:
    if response_format is None:
        return None
    json_schema = getattr(response_format, "json_schema", None)
    if isinstance(json_schema, dict):
        return {
            "format": {
                "type": "json_schema",
                "name": getattr(response_format, "name", None) or "response",
                "schema": json_schema,
                "strict": bool(getattr(response_format, "strict", False)),
            }
        }
    return {"format": {"type": "text"}}


def _extract_text_from_response_payload(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
            text = first.get("text")
            if isinstance(text, str):
                return text

    output = payload.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                text = content_item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)

    content = payload.get("content")
    if isinstance(content, str):
        return content
    return ""


def _extract_delta_from_stream_payload(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    event_type = payload.get("type")
    if isinstance(event_type, str) and event_type.endswith(".delta"):
        delta = payload.get("delta")
        if isinstance(delta, str):
            return delta

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            delta = first.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    return content
            text = first.get("text")
            if isinstance(text, str):
                return text
    return ""


class CustomCompletionPathClient:
    """Minimal OpenAI-compatible client honoring CUSTOM_LLM_COMPLETION_PATH."""

    def __init__(self, endpoint: str):
        self.base_url = endpoint
        self.completion_url = endpoint

    @property
    def _uses_responses_endpoint(self) -> bool:
        path = urlsplit(self.completion_url).path.rstrip("/").lower()
        return path.endswith("/responses")

    def generate(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            if self._uses_responses_endpoint:
                return self._generate_once_as_stream(kwargs)
            return self._generate_stream(kwargs)
        return self._generate_once(kwargs)

    def _headers(self) -> dict[str, str]:
        api_key = get_custom_llm_api_key_env() or "null"
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, kwargs: dict[str, Any], *, stream: bool) -> dict[str, Any]:
        messages = _messages_to_payload_messages(list(kwargs.get("messages") or []))
        payload: dict[str, Any]

        if self._uses_responses_endpoint:
            payload = {
                "model": kwargs.get("model"),
                "input": messages,
                "stream": stream,
            }
            response_format = _response_format_for_responses(kwargs.get("response_format"))
            if response_format:
                payload["text"] = response_format
        else:
            payload = {
                "model": kwargs.get("model"),
                "messages": messages,
                "stream": stream,
            }
            response_format = _response_format_for_chat(kwargs.get("response_format"))
            if response_format:
                payload["response_format"] = response_format

        if kwargs.get("max_tokens") is not None:
            payload["max_tokens"] = kwargs["max_tokens"]
        extra_body = kwargs.get("extra_body")
        if isinstance(extra_body, dict):
            payload.update(extra_body)
        return payload

    def _generate_once(self, kwargs: dict[str, Any]) -> Any:
        payload = self._build_payload(kwargs, stream=False)
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                self.completion_url,
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            body = response.json()

        text = _extract_text_from_response_payload(body)
        return SimpleNamespace(content=text, tool_calls=[], messages=[])

    def _generate_once_as_stream(self, kwargs: dict[str, Any]) -> Iterator[Any]:
        response = self._generate_once(kwargs)
        text = response.content if isinstance(response.content, str) else ""
        if text:
            yield SimpleNamespace(type="content", chunk=text)
        yield SimpleNamespace(type="completion", content=text)

    def _generate_stream(self, kwargs: dict[str, Any]) -> Iterator[Any]:
        payload = self._build_payload(kwargs, stream=True)
        accumulated: list[str] = []
        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST",
                self.completion_url,
                headers=self._headers(),
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError:
                        LOGGER.debug("Skipping non-JSON custom stream event: %s", data)
                        continue
                    delta = _extract_delta_from_stream_payload(parsed)
                    if delta:
                        accumulated.append(delta)
                        yield SimpleNamespace(type="content", chunk=delta)

        yield SimpleNamespace(type="completion", content="".join(accumulated))


def get_custom_completion_path_client() -> Optional[CustomCompletionPathClient]:
    endpoint = build_custom_completion_endpoint(
        get_custom_llm_url_env(),
        get_custom_llm_completion_path_env(),
    )
    if not endpoint:
        return None
    return CustomCompletionPathClient(endpoint)

import httpx
from llmai.shared import JSONSchemaResponse, SystemMessage, UserMessage

from utils.custom_completion_path_client import (
    CustomCompletionPathClient,
    build_custom_completion_endpoint,
)


def test_build_custom_completion_endpoint_joins_base_and_path():
    assert (
        build_custom_completion_endpoint(
            "https://api.with7.cn", "/chatgpt/v1/responses"
        )
        == "https://api.with7.cn/chatgpt/v1/responses"
    )


def test_custom_completion_path_client_posts_to_responses_endpoint(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = request.read().decode("utf-8")
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": '{"slides":[]}'}
                        ],
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)

    def fake_client(*args, **kwargs):
        return httpx.Client(transport=transport, timeout=kwargs.get("timeout"))

    monkeypatch.setenv("CUSTOM_LLM_API_KEY", "test-key")
    monkeypatch.setattr("utils.custom_completion_path_client.httpx.Client", fake_client)

    client = CustomCompletionPathClient(
        "https://api.with7.cn/chatgpt/v1/responses"
    )
    events = list(client.generate(
        model="gpt-5.5",
        messages=[
            SystemMessage(content="Return JSON."),
            UserMessage(content="Make slides."),
        ],
        response_format=JSONSchemaResponse(
            name="response",
            json_schema={
                "type": "object",
                "properties": {"slides": {"type": "array"}},
                "required": ["slides"],
                "additionalProperties": False,
            },
            strict=True,
        ),
        stream=True,
    ))

    assert captured["url"] == "https://api.with7.cn/chatgpt/v1/responses"
    assert "Bearer test-key" == captured["headers"]["authorization"]
    assert events[0].type == "content"
    assert events[0].chunk == '{"slides":[]}'
    assert events[-1].type == "completion"
    assert '"input"' in captured["body"]
    assert '"text"' in captured["body"]
    assert '"stream":false' in captured["body"].replace(" ", "")

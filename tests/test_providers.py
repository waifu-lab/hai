import asyncio
import json

import httpx
import pytest

from hai.core import Flow
from hai.providers import (
    Anthropic,
    AnthropicText,
    GoogleText,
    OpenAIEmbedding,
    OpenAIText,
    ToolLoopLimitError,
    create_text_provider,
)
from hai.tool import tool


@pytest.fixture(autouse=True)
def no_api_keys(monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def chat_response(content="hi", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return httpx.Response(
        200,
        json={
            "choices": [{"message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


def test_missing_api_key_names_the_env_var():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIText()
    with pytest.raises(ValueError, match="GEMINI_API_KEY or GOOGLE_API_KEY"):
        GoogleText()


def test_api_key_read_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-env")
    assert OpenAIText().api_key == "sk-env"
    assert create_text_provider("openai", model="gpt-4o-mini").api_key == "sk-env"
    assert GoogleText().api_key == "g-env"
    assert OpenAIText(api_key="sk-arg").api_key == "sk-arg"


def test_base_url_override(mock_http):
    mock_http.handler = lambda _req: chat_response("from proxy")
    llm = OpenAIText(api_key="k", base_url="https://proxy.example/v1/chat/completions")
    context = asyncio.run(Flow(llm).run({"messages": [{"role": "user", "content": "hi"}]}))
    assert context["assistant_message"] == "from proxy"
    assert str(mock_http.requests[0].url) == "https://proxy.example/v1/chat/completions"
    assert AnthropicText(api_key="k", base_url="https://a.example").base_url == "https://a.example"


def test_tool_loop_stops_at_max_rounds(mock_http):
    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    call = [{"id": "c1", "type": "function", "function": {"name": "ping", "arguments": "{}"}}]
    mock_http.handler = lambda _req: chat_response("", tool_calls=call)
    llm = OpenAIText(api_key="k", max_tool_rounds=2)
    with pytest.raises(ToolLoopLimitError):
        asyncio.run(llm.exec({"messages": [{"role": "user", "content": "hi"}], "tools": [ping]}))
    # Two allowed rounds, the third request triggers the limit
    assert len(mock_http.requests) == 3


def test_streaming_node_passes_chunks_to_on_chunk(mock_http):
    events = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    mock_http.handler = lambda _req: httpx.Response(200, text=body)

    chunks = []
    llm = OpenAIText(api_key="k", stream=True)
    context = asyncio.run(llm_run(llm, chunks.append))
    assert chunks == ["Hel", "lo"]
    assert context["assistant_message"] == "Hello"


async def llm_run(llm, on_chunk):
    context = {"messages": [{"role": "user", "content": "hi"}], "on_chunk": on_chunk}
    await llm.exec(context)
    return context


def test_embedding_retries_on_5xx(mock_http):
    responses = iter([httpx.Response(503), httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1]}]})])
    mock_http.handler = lambda _req: next(responses)
    context = {"input": "hello"}
    asyncio.run(OpenAIEmbedding(api_key="k").exec(context))
    assert context["embeddings"] == [[0.1]]
    assert len(mock_http.requests) == 2


def test_anthropic_models_uses_native_headers_and_pages(mock_http):
    pages = iter(
        [
            {"data": [{"id": "claude-a"}], "has_more": True, "last_id": "claude-a"},
            {"data": [{"id": "claude-b"}], "has_more": False, "last_id": "claude-b"},
        ]
    )
    mock_http.handler = lambda _req: httpx.Response(200, json=next(pages))
    models = asyncio.run(Anthropic(api_key="ak").models())
    assert models == ["claude-a", "claude-b"]
    first, second = mock_http.requests
    assert first.headers["x-api-key"] == "ak"
    assert "anthropic-version" in first.headers
    assert "authorization" not in first.headers
    assert second.url.params["after_id"] == "claude-a"

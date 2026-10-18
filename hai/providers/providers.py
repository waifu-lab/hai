import asyncio
import email.utils
import inspect
import json
import logging
import os
import time
import warnings
from abc import abstractmethod
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Sequence, Union

import httpx
from httpx import AsyncClient
from pydantic import BaseModel

from hai.core import BaseNode
from hai.tool import Tool, run_tool_calls

logger = logging.getLogger(__name__)


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Seconds to wait from a ``Retry-After`` header (delta-seconds or HTTP date)."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_at = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, retry_at.timestamp() - time.time())


# RemoteProtocolError: the server closed the connection mid-response (common while streaming)
_TRANSPORT_ERRORS = (httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError, httpx.HTTPStatusError)


class StreamInterruptedError(Exception):
    """A streaming turn failed after some content had already been passed to ``on_chunk``.

    Not retried here: a retry would re-send text the caller has already shown.
    ``partial_content`` is what was streamed before the failure; the caller decides
    whether to discard it and run the turn again.
    """

    def __init__(self, partial_content: str, cause: Exception) -> None:
        super().__init__(f"Stream interrupted after {len(partial_content)} chars: {cause}")
        self.partial_content = partial_content
        self.cause = cause


def _is_retryable(exc: Exception) -> bool:
    """Network errors, timeouts, 429 and 5xx are retried; other HTTP errors are not."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, (httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError))


def _retry_delay(attempt: int, exc: Exception) -> float:
    """Exponential backoff. 429 honours ``Retry-After`` and otherwise backs off harder."""
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        retry_after = _parse_retry_after(exc.response.headers.get("retry-after"))
        return retry_after if retry_after is not None else float(5 * 2**attempt)
    return float(2**attempt)


def resolve_api_key(api_key: Optional[str], env_vars: Sequence[str], provider: str) -> str:
    """Return *api_key*, or the first of *env_vars* that is set, raising a clear error when neither is."""
    if api_key:
        return api_key
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            return value
    raise ValueError(
        f"No API key for {provider}. Pass api_key=... or set the {' or '.join(env_vars)} environment variable."
    )


async def _post_json_with_retry(
    client: AsyncClient,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    max_retries: int,
) -> Any:
    """POST *payload* and return the JSON body, with exponential backoff on network errors, 429 and 5xx."""
    attempts = max(1, max_retries)
    for attempt in range(attempts):
        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except _TRANSPORT_ERRORS as e:
            if not _is_retryable(e) or attempt == attempts - 1:
                raise
            await asyncio.sleep(_retry_delay(attempt, e))
    raise AssertionError("unreachable")


class ToolLoopLimitError(RuntimeError):
    """The model kept requesting tool calls past ``max_tool_rounds``."""


def _tool_calls_from_buffer(tool_calls_buffer: Dict[int, Dict[str, str]]) -> List[Dict[str, Any]]:
    """Turn streamed tool-call fragments into OpenAI-format tool calls, in index order."""
    return [
        {
            "id": tool_calls_buffer[i]["id"],
            "type": "function",
            "function": {
                "name": tool_calls_buffer[i]["name"],
                "arguments": tool_calls_buffer[i]["arguments"],
            },
        }
        for i in sorted(tool_calls_buffer)
    ]


class TurnResult(BaseModel):
    """Single-turn LLM response. Tool calls are in OpenAI format (not executed)."""

    content: str
    tool_calls: List[Dict[str, Any]]
    finish_reason: str
    usage: Dict[str, int]


class ProviderInfo:
    """Provider-level operations such as listing models. Not a pipeline node."""

    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
    ):
        self.name = name
        self.api_key = api_key
        self.base_url = base_url

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    async def models(self) -> List[str]:
        async with AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )
            response.raise_for_status()
            res = response.json()
            return [item["id"] for item in res.get("data", [])]


class ProviderText(BaseNode):
    """Chat completion node for any OpenAI-compatible ``/chat/completions`` endpoint.

    As a node it runs the full tool loop: tool calls are executed and fed back
    until the model returns a final answer. Use :meth:`call_once` to drive the
    loop yourself.

    The tool loop stops with :class:`ToolLoopLimitError` after ``max_tool_rounds``
    rounds of tool calls, so a model that never stops calling tools cannot run forever.

    Context:
        reads: ``messages`` (List[dict]), ``tools`` (List[Tool], optional),
            ``on_chunk`` (callable, optional; receives each text chunk when ``stream=True``)
        writes: ``messages`` (conversation including tool-call turns),
            ``assistant_message`` (str), ``openai_response`` (raw response, non-stream),
            ``parsed_response`` / ``parse_error`` (when ``response_format`` is ``json_object``)
    """

    _max_tokens_field = "max_tokens"
    """Request field that carries ``max_tokens``; subclasses override it when the API names it differently."""

    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        model: str,
        temperature: Optional[float],
        max_tokens: Optional[int] = None,
        stream: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        max_tool_rounds: int = 10,
    ):
        super().__init__(name)
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.base_url = base_url
        self.response_format = response_format
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_tool_rounds = max_tool_rounds

    def _extra_payload(self) -> Dict[str, Any]:
        """Subclasses override to inject provider-specific payload fields (e.g. thinking, reasoning_effort)."""
        return {}

    async def exec(self, context: Dict[str, Any]) -> Any:
        messages = context.get("messages", [])
        tools: List[Tool] = context.get("tools", [])

        if not messages:
            raise ValueError("'messages' is required in context")

        payload = {**self._build_payload(messages, tools), "stream": self.stream}

        async with AsyncClient(timeout=self.timeout) as client:
            if self.stream:
                return await self._stream_request(client, self._headers(), payload, context, tools)
            else:
                return await self._request(client, self._headers(), payload, context, tools)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, messages: List[Dict[str, Any]], tools: Optional[List[Tool]]) -> Dict[str, Any]:
        """Request body shared by :meth:`exec` and :meth:`call_once`, without ``stream``.

        ``temperature`` is omitted when None so the provider default applies.
        """
        payload: Dict[str, Any] = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens:
            payload[self._max_tokens_field] = self.max_tokens
        if self.response_format:
            payload["response_format"] = self.response_format
        if tools:
            payload["tools"] = [t.to_dict() for t in tools]
            payload["tool_choice"] = "auto"
        payload.update(self._extra_payload())
        return payload

    async def _post_with_retry(
        self,
        client: AsyncClient,
        headers: Dict[str, str],
        payload: Dict[str, Any],
    ) -> Any:
        """POST with exponential backoff on network errors, 429 and 5xx responses."""
        return await _post_json_with_retry(client, self.base_url, headers, payload, self.max_retries)

    def _check_tool_round(self, rounds: int) -> None:
        if rounds > self.max_tool_rounds:
            raise ToolLoopLimitError(
                f"{self.name}: model still requested tools after {self.max_tool_rounds} rounds; "
                "raise max_tool_rounds or check the tools' results."
            )

    async def _request(
        self,
        client: AsyncClient,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        context: Dict[str, Any],
        tools: Optional[List[Tool]] = None,
    ) -> str:
        messages: List[Dict[str, Any]] = list(payload["messages"])
        tool_rounds = 0

        while True:
            current_payload = {**payload, "messages": messages}
            data = await self._post_with_retry(client, headers, current_payload)

            if not data.get("choices"):
                raise ValueError(f"API returned no choices: {data}")

            choice = data["choices"][0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason")

            if finish_reason == "tool_calls" and tools:
                tool_rounds += 1
                self._check_tool_round(tool_rounds)
                messages.append(message)
                messages.extend(await run_tool_calls(message.get("tool_calls", []), tools))
            else:
                content = message.get("content", "")
                context["openai_response"] = data
                context["assistant_message"] = content
                context["messages"] = messages

                if self.response_format and self.response_format.get("type") == "json_object":
                    try:
                        parsed_content = json.loads(content)
                        context["parsed_response"] = parsed_content
                        return parsed_content
                    except json.JSONDecodeError:
                        context["parse_error"] = "Failed to parse JSON response"

                return content

    async def _stream_one_turn(  # noqa: C901
        self,
        client: AsyncClient,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        on_chunk: Optional[Any] = None,
    ):
        """Stream one LLM turn with retry on network errors, 429 and 5xx.

        Retries only while nothing has been passed to ``on_chunk``; a failure after that
        raises :class:`StreamInterruptedError` instead.

        Returns (full_content, tool_calls_buffer, finish_reason, usage).
        usage dict has keys "input", "output", "total"; zeros if not provided by API.
        """
        merged_payload = {**payload, "stream_options": {"include_usage": True}}

        attempts = max(1, self.max_retries)
        for attempt in range(attempts):
            full_content = ""
            tool_calls_buffer: Dict[int, Dict[str, str]] = {}
            finish_reason = None
            usage: Dict[str, int] = {"input": 0, "output": 0, "total": 0}
            done_received = False
            try:
                async with client.stream("POST", self.base_url, headers=headers, json=merged_payload) as response:
                    if response.status_code >= 500:
                        response.raise_for_status()
                    if response.status_code >= 400:
                        body = await response.aread()
                        # TODO: Replace with a custom error
                        logger.error(
                            "HTTP %s from %s: %s",
                            response.status_code,
                            self.base_url,
                            body.decode(errors="replace"),
                        )
                        response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if data_str == "[DONE]":
                            done_received = True
                            break

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        if "usage" in data and not data.get("choices"):
                            raw = data["usage"]
                            usage = {
                                "input": raw.get("prompt_tokens", 0),
                                "output": raw.get("completion_tokens", 0),
                                "total": raw.get("total_tokens", 0),
                            }
                            continue

                        if not data.get("choices"):
                            continue

                        choice = data["choices"][0]
                        delta = choice.get("delta", {})
                        finish_reason = choice.get("finish_reason") or finish_reason

                        if "content" in delta and delta["content"]:
                            chunk = delta["content"]
                            full_content += chunk
                            if on_chunk is not None:
                                if inspect.iscoroutinefunction(on_chunk):
                                    await on_chunk(chunk)
                                else:
                                    on_chunk(chunk)

                        for tc_delta in delta.get("tool_calls", []):
                            idx = tc_delta.get("index", 0)
                            if idx not in tool_calls_buffer:
                                tool_calls_buffer[idx] = {"id": "", "name": "", "arguments": ""}
                            if "id" in tc_delta:
                                tool_calls_buffer[idx]["id"] += tc_delta["id"]
                            func = tc_delta.get("function", {})
                            if "name" in func:
                                tool_calls_buffer[idx]["name"] += func["name"]
                            if "arguments" in func:
                                tool_calls_buffer[idx]["arguments"] += func["arguments"]

                if not done_received and not tool_calls_buffer:
                    warnings.warn("Stream ended before [DONE] was received; response may be incomplete.", stacklevel=2)

                return full_content, tool_calls_buffer, finish_reason, usage

            except _TRANSPORT_ERRORS as e:
                if full_content and on_chunk is not None:
                    raise StreamInterruptedError(full_content, e) from e
                if not _is_retryable(e) or attempt == attempts - 1:
                    raise
                await asyncio.sleep(_retry_delay(attempt, e))

        raise AssertionError("unreachable")

    async def _stream_request(
        self,
        client: AsyncClient,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        context: Dict[str, Any],
        tools: Optional[List[Tool]] = None,
    ) -> str:
        messages: List[Dict[str, Any]] = list(payload["messages"])
        on_chunk = context.get("on_chunk")
        tool_rounds = 0

        while True:
            current_payload = {**payload, "messages": messages}
            full_content, tool_calls_buffer, finish_reason, _ = await self._stream_one_turn(
                client, headers, current_payload, on_chunk=on_chunk
            )

            if finish_reason == "tool_calls" and tools and tool_calls_buffer:
                tool_rounds += 1
                self._check_tool_round(tool_rounds)
                tool_calls_list = _tool_calls_from_buffer(tool_calls_buffer)
                messages.append(
                    {
                        "role": "assistant",
                        "content": full_content or None,
                        "tool_calls": tool_calls_list,
                    }
                )
                messages.extend(await run_tool_calls(tool_calls_list, tools))
            else:
                context["assistant_message"] = full_content
                context["messages"] = messages
                return full_content

    async def call_once(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Tool]] = None,
        system_prompt: Optional[str] = None,
        on_chunk: Optional[Any] = None,
    ) -> TurnResult:
        """Single LLM API call — no tool execution.

        Prepends system_prompt as a system message if provided.
        Returns TurnResult with tool_calls in normalized OpenAI format.
        The Agent drives the tool loop; this method only covers one turn.
        Use :func:`hai.tool.run_tool_calls` to execute the returned tool calls.
        """
        full_messages: List[Dict[str, Any]] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload = self._build_payload(full_messages, tools)
        headers = self._headers()

        async with AsyncClient(timeout=self.timeout) as client:
            if self.stream:
                content, tool_calls_buffer, finish_reason, usage = await self._stream_one_turn(
                    client, headers, {**payload, "stream": True}, on_chunk=on_chunk
                )
                tool_calls = _tool_calls_from_buffer(tool_calls_buffer)
            else:
                data = await self._post_with_retry(client, headers, {**payload, "stream": False})
                if not data.get("choices"):
                    raise ValueError(f"API returned no choices: {data}")
                choice = data["choices"][0]
                message = choice["message"]
                finish_reason = choice.get("finish_reason", "") or ""
                content = message.get("content", "") or ""
                tool_calls = message.get("tool_calls", []) or []
                raw_usage = data.get("usage", {})
                usage = {
                    "input": raw_usage.get("prompt_tokens", 0),
                    "output": raw_usage.get("completion_tokens", 0),
                    "total": raw_usage.get("total_tokens", 0),
                }

        return TurnResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason or "",
            usage=usage,
        )


class ProviderSTT(BaseNode):
    """Speech-to-text node for OpenAI-compatible ``/audio/transcriptions`` endpoints.

    Context:
        reads: ``audio`` (bytes), ``format`` (str, default ``"webm"``), ``language`` (str, optional)
        writes: ``transcript`` (str)
    """

    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        model: str = "whisper-1",
        language: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(name)
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.language = language
        self.temperature = temperature
        self.timeout = timeout

    async def exec(self, context: Dict[str, Any]) -> Any:
        audio: bytes = context["audio"]
        fmt: str = context.get("format", "webm")
        language: Optional[str] = context.get("language", self.language)
        transcript = await self.transcribe(audio, fmt, language)
        context["transcript"] = transcript
        return transcript

    async def transcribe(
        self,
        audio: bytes,
        format: str = "webm",
        language: Optional[str] = None,
    ) -> str:
        """Convert audio bytes to text."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {"file": (f"audio.{format}", audio, f"audio/{format}")}
        data: Dict[str, Any] = {
            "model": self.model,
            "response_format": "text",
            "temperature": str(self.temperature),
        }
        if language or self.language:
            data["language"] = language or self.language

        async with AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.base_url,
                headers=headers,
                files=files,
                data=data,
            )
            response.raise_for_status()
            return response.text.strip()


class ProviderTTS(BaseNode):
    """Base class for text-to-speech nodes.

    Subclasses implement :meth:`synthesize` and :meth:`synthesize_stream` with their
    own provider-specific options; the node behaviour is shared.

    Context:
        reads: ``speech_text`` (str)
        writes: ``speech_audio`` (bytes)
    """

    def __init__(self, name: str, base_url: str, timeout: float = 60.0) -> None:
        super().__init__(name)
        self.base_url = base_url
        self.timeout = timeout

    async def exec(self, context: Dict[str, Any]) -> Any:
        audio = await self.synthesize(context["speech_text"])
        context["speech_audio"] = audio
        return audio

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Convert text to complete audio bytes."""

    @abstractmethod
    def synthesize_stream(self, text: str, chunk_size: int = 4096) -> AsyncIterator[bytes]:
        """Yield audio bytes in chunks as they are produced."""


class ProviderEmbedding(BaseNode):
    """Embedding node for OpenAI-compatible ``/embeddings`` endpoints.

    Context:
        reads: ``input`` (str or List[str])
        writes: ``embeddings`` (List[List[float]], one per input; a single str gives a
            one-element list), ``response`` (raw API response)
    """

    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        model: str,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        super().__init__(name)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout

    async def exec(self, context: Dict[str, Any]) -> Any:
        input_text = context.get("input")

        if not input_text:
            raise ValueError("'input' is required in context for embedding generation")

        payload = {
            "model": self.model,
            "input": input_text,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with AsyncClient(timeout=self.timeout) as client:
            data = await _post_json_with_retry(client, self.base_url, headers, payload, self.max_retries)

        embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda item: item.get("index", 0))]
        context["embeddings"] = embeddings
        context["response"] = data
        return embeddings

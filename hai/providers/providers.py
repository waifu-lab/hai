import asyncio
import base64
import inspect
import warnings
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union

import httpx
from httpx import AsyncClient
from pydantic import BaseModel

from hai.core import BaseNode
from hai.tool import Tool

# curl https://api.openai.com/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -d '{
#     "model": "gpt-5",
#     "messages": [
#       {
#         "role": "developer",
#         "content": "You are a helpful assistant."
#       },
#       {
#         "role": "user",
#         "content": "Hello!"
#       }
#     ]
#   }'

# {
#   "id": "chatcmpl-B9MBs8CjcvOU2jLn4n570S5qMJKcT",
#   "object": "chat.completion",
#   "created": 1741569952,
#   "model": "gpt-4.1-2025-04-14",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "Hello! How can I assist you today?",
#         "refusal": null,
#         "annotations": []
#       },
#       "logprobs": null,
#       "finish_reason": "stop"
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 19,
#     "completion_tokens": 10,
#     "total_tokens": 29,
#     "prompt_tokens_details": {
#       "cached_tokens": 0,
#       "audio_tokens": 0
#     },
#     "completion_tokens_details": {
#       "reasoning_tokens": 0,
#       "audio_tokens": 0,
#       "accepted_prediction_tokens": 0,
#       "rejected_prediction_tokens": 0
#     }
#   },
#   "service_tier": "default"
# }


class TurnResult(BaseModel):
    """Single-turn LLM response. Tool calls are in OpenAI format (not executed)."""

    content: str
    tool_calls: List[Dict[str, Any]]
    finish_reason: str
    usage: Dict[str, int]


class ProviderInfo(BaseNode):
    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
    ):
        super().__init__(name)
        self.api_key = api_key
        self.base_url = base_url

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
    """
    OpenAI
    Anthropic
    Google
    ...
    """

    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
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

    def _extra_payload(self) -> Dict[str, Any]:
        """Subclasses override to inject provider-specific payload fields (e.g. thinking, reasoning_effort)."""
        return {}

    async def exec(self, context: Dict[str, Any]) -> Any:
        messages = context.get("messages", [])
        tools: List[Tool] = context.get("tools", [])

        if not messages:
            raise ValueError("context 中必須包含 'messages'")

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": self.stream,
        }

        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        if self.response_format:
            payload["response_format"] = self.response_format

        if tools:
            payload["tools"] = [t.to_dict() for t in tools]
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with AsyncClient(timeout=self.timeout) as client:
            if self.stream:
                return await self._stream_request(client, headers, payload, context, tools)
            else:
                return await self._request(client, headers, payload, context, tools)

    async def _post_with_retry(
        self,
        client: AsyncClient,
        headers: Dict[str, str],
        payload: Dict[str, Any],
    ) -> Any:
        """POST with exponential backoff retry on network errors and 5xx responses."""
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = await client.post(self.base_url, headers=headers, json=payload)
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Server error {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response.json()
            except (httpx.NetworkError, httpx.TimeoutException) as e:
                last_exc = e
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    last_exc = e
                elif e.response.status_code == 429:
                    last_exc = e  # TODO:
                    raise
                else:
                    raise
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2**attempt)
        raise last_exc

    async def _request(
        self,
        client: AsyncClient,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        context: Dict[str, Any],
        tools: Optional[List[Tool]] = None,
    ) -> str:
        import json

        tool_map = {t.name: t for t in (tools or [])}
        messages: List[Dict[str, Any]] = list(payload["messages"])

        while True:
            current_payload = {**payload, "messages": messages}
            data = await self._post_with_retry(client, headers, current_payload)

            if not data.get("choices"):
                raise ValueError(f"API returned no choices: {data}")

            choice = data["choices"][0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason")

            if finish_reason == "tool_calls" and tool_map:
                messages.append(message)

                for tool_call in message.get("tool_calls", []):
                    func_name = tool_call["function"]["name"]
                    func_args = tool_call["function"]["arguments"]
                    tool_id = tool_call["id"]

                    if func_name in tool_map:
                        result = await tool_map[func_name].call(func_args)
                    else:
                        result = f"Tool '{func_name}' not found"

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": result,
                        }
                    )
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

    async def _stream_one_turn(
        self,
        client: AsyncClient,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        on_chunk: Optional[Any] = None,
    ):
        """Stream one LLM turn with retry on network errors and 5xx.

        Returns (full_content, tool_calls_buffer, finish_reason, usage).
        usage dict has keys "input", "output", "total"; zeros if not provided by API.
        """
        import json

        merged_payload = {**payload, "stream_options": {"include_usage": True}}

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            full_content = ""
            tool_calls_buffer: Dict[int, Dict[str, str]] = {}
            finish_reason = None
            usage: Dict[str, int] = {"input": 0, "output": 0, "total": 0}
            done_received = False
            try:
                async with client.stream("POST", self.base_url, headers=headers, json=merged_payload) as response:
                    if response.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            f"Server error {response.status_code}",
                            request=response.request,
                            response=response,
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

            except (httpx.NetworkError, httpx.TimeoutException) as e:
                last_exc = e
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 or e.response.status_code >= 500:
                    if e.response.status_code == 429:
                        pass  # TODO:
                    last_exc = e
                else:
                    raise

            if attempt < self.max_retries - 1:
                delay = 2**attempt
                if isinstance(last_exc, httpx.HTTPStatusError) and last_exc.response.status_code == 429:
                    retry_after = last_exc.response.headers.get("retry-after")
                    delay = float(retry_after) if retry_after else float(5 * (2**attempt))
                await asyncio.sleep(delay)

        raise last_exc

    async def _stream_request(
        self,
        client: AsyncClient,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        context: Dict[str, Any],
        tools: Optional[List[Tool]] = None,
    ) -> str:
        tool_map = {t.name: t for t in (tools or [])}
        messages: List[Dict[str, Any]] = list(payload["messages"])

        while True:
            current_payload = {**payload, "messages": messages}
            full_content, tool_calls_buffer, finish_reason, _ = await self._stream_one_turn(
                client, headers, current_payload
            )

            if full_content:
                ...

            if finish_reason == "tool_calls" and tool_map and tool_calls_buffer:
                tool_calls_list = [
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
                messages.append(
                    {
                        "role": "assistant",
                        "content": full_content or None,
                        "tool_calls": tool_calls_list,
                    }
                )

                for tool_call in tool_calls_list:
                    func_name = tool_call["function"]["name"]
                    func_args = tool_call["function"]["arguments"]
                    tool_id = tool_call["id"]

                    if func_name in tool_map:
                        result = await tool_map[func_name].call(func_args)
                    else:
                        result = f"Tool '{func_name}' not found"

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": result,
                        }
                    )
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
        """
        full_messages: List[Dict[str, Any]] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": full_messages,
            "temperature": self.temperature,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if self.response_format:
            payload["response_format"] = self.response_format
        if tools:
            payload["tools"] = [t.to_dict() for t in tools]
            payload["tool_choice"] = "auto"
        payload.update(self._extra_payload())

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with AsyncClient(timeout=self.timeout) as client:
            if self.stream:
                content, tool_calls_buffer, finish_reason, usage = await self._stream_one_turn(
                    client, headers, {**payload, "stream": True}, on_chunk=on_chunk
                )
                if content:
                    ...
                tool_calls = [
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
        text = await self.transcribe(audio, fmt, language)
        context["text"] = text
        return text

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
    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        model: str = "tts-1",
        voice: str = "nova",
        response_format: str = "mp3",
        speed: float = 1.0,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(name)
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.voice = voice
        self.response_format = response_format
        self.speed = speed
        self.timeout = timeout

    async def exec(self, context: Dict[str, Any]) -> Any:
        text: str = context["text"]
        audio = await self.synthesize(text)
        context["audio"] = audio
        return audio

    async def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes (default is mp3)."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": self.response_format,
            "speed": self.speed,
        }
        async with AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.content

    async def synthesize_stream(self, text: str, chunk_size: int = 4096) -> "AsyncIterator[bytes]":
        """Stream audio synthesis."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": self.response_format,
            "speed": self.speed,
        }
        async with AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", self.base_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size):
                    if chunk:
                        yield chunk


class ProviderEmbedding(BaseNode):
    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        model: str,
    ):
        super().__init__(name)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

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

        async with AsyncClient(timeout=30.0) as client:
            response = await client.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()

            if isinstance(input_text, list):
                embeddings = [item["embedding"] for item in data["data"]]
                context["embeddings"] = embeddings
                return embeddings
            else:
                embedding = data["data"][0]["embedding"]
                context["embedding"] = embedding
                context["response"] = data
                return embedding

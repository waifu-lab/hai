from typing import Any, AsyncIterator, Dict, Literal, Optional, Tuple

from httpx import AsyncClient

from .providers import ProviderEmbedding, ProviderInfo, ProviderSTT, ProviderText, ProviderTTS

_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_MAX_TOKENS = 1500


class OpenAI(ProviderInfo):
    def __init__(
        self,
        name: str = "OpenAI",
        api_key: str = "your_openai_api_key",
        base_url: str = "https://api.openai.com/v1",
    ):
        super().__init__(
            name=name,
            api_key=api_key,
            base_url=base_url,
        )


class OpenAIText(ProviderText):
    """OpenAI Chat Completions.

    The output limit is sent as ``max_completion_tokens``, which every chat model
    accepts; reasoning models (o-series, gpt-5) reject the legacy ``max_tokens``.

    Set ``reasoning_effort`` when using a reasoning model. Those models only accept
    the default temperature, so ``temperature`` is then left unset (passing anything
    other than 1 raises), and reasoning tokens count against the output limit, so
    ``max_tokens`` is left unset too unless given, letting the model use its own
    maximum. Without ``reasoning_effort`` the defaults are ``temperature=0.7`` and
    ``max_tokens=1500``.
    """

    _max_tokens_field = "max_completion_tokens"

    def __init__(
        self,
        name: str = "OpenAI",
        api_key: str = "your_openai_api_key",
        model: str = "gpt-4o-mini",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        max_retries: int = 3,
        timeout: float = 60.0,
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ):
        if reasoning_effort is None:
            temperature = _DEFAULT_TEMPERATURE if temperature is None else temperature
            max_tokens = _DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
        elif temperature is not None and temperature != 1:
            raise ValueError(f"Reasoning models only support temperature=1, got {temperature}.")

        super().__init__(
            name=name,
            api_key=api_key,
            model=model,
            base_url="https://api.openai.com/v1/chat/completions",
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            max_retries=max_retries,
            timeout=timeout,
        )
        self.reasoning_effort = reasoning_effort
        self.extra_params = extra_params or {}

    def _extra_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        payload.update(self.extra_params)
        return payload


class OpenAIEmbedding(ProviderEmbedding):
    def __init__(
        self,
        name: str = "OpenAIEmbedding",
        api_key: str = "your_openai_api_key",
        model: str = "text-embedding-3-small",
    ):
        super().__init__(
            name=name,
            api_key=api_key,
            model=model,
            base_url="https://api.openai.com/v1/embeddings",
        )


class OpenAISTT(ProviderSTT):
    def __init__(
        self,
        name: str = "OpenAISTT",
        api_key: str = "your_openai_api_key",
        model: str = "whisper-1",
        language: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ):
        super().__init__(
            name=name,
            api_key=api_key,
            base_url="https://api.openai.com/v1/audio/transcriptions",
            model=model,
            language=language,
            temperature=temperature,
            timeout=timeout,
        )


class OpenAITTS(ProviderTTS):
    """Text-to-speech through an OpenAI-compatible ``/audio/speech`` endpoint."""

    def __init__(
        self,
        name: str = "OpenAITTS",
        api_key: str = "your_openai_api_key",
        model: str = "tts-1",
        voice: str = "nova",
        response_format: str = "mp3",
        speed: float = 1.0,
        timeout: float = 60.0,
        base_url: str = "https://api.openai.com/v1/audio/speech",
    ):
        super().__init__(name=name, base_url=base_url, timeout=timeout)
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.response_format = response_format
        self.speed = speed

    def _request(self, text: str) -> Tuple[Dict[str, str], Dict[str, Any]]:
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
        return headers, payload

    async def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes in :attr:`response_format` (default mp3)."""
        headers, payload = self._request(text)
        async with AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.content

    async def synthesize_stream(self, text: str, chunk_size: int = 4096) -> AsyncIterator[bytes]:
        """Stream audio synthesis."""
        headers, payload = self._request(text)
        async with AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", self.base_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size):
                    if chunk:
                        yield chunk

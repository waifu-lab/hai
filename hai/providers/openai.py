from typing import Any, Dict, Literal, Optional

from .providers import ProviderEmbedding, ProviderInfo, ProviderSTT, ProviderText, ProviderTTS


class OpenAI(ProviderInfo):
    def __init__(self, name: str = "OpenAI"):
        super().__init__(
            name=name,
            api_key="your_openai_api_key",
            base_url="https://api.openai.com/v1",
        )


class OpenAIText(ProviderText):
    def __init__(
        self,
        name: str = "OpenAI",
        api_key: str = "your_openai_api_key",
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1500,
        stream: bool = False,
        max_retries: int = 3,
        timeout: float = 60.0,
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
    ):
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

    def _extra_payload(self) -> Dict[str, Any]:
        if self.reasoning_effort is not None:
            return {"reasoning_effort": self.reasoning_effort}
        return {}


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
        language: str | None = None,
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
    def __init__(
        self,
        name: str = "OpenAITTS",
        api_key: str = "your_openai_api_key",
        model: str = "tts-1",
        voice: str = "nova",
        response_format: str = "mp3",
        speed: float = 1.0,
        timeout: float = 60.0,
    ):
        super().__init__(
            name=name,
            api_key=api_key,
            base_url="https://api.openai.com/v1/audio/speech",
            model=model,
            voice=voice,
            response_format=response_format,
            speed=speed,
            timeout=timeout,
        )

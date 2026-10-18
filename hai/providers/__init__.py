from typing import Any, Dict, Literal, Optional, Type

from hai.providers.anthropic import Anthropic, AnthropicText
from hai.providers.google import Google, GoogleEmbedding, GoogleText
from hai.providers.gptsovits import GPTSoVITS, GPTSoVITSTTS
from hai.providers.ollama import Ollama, OllamaText
from hai.providers.openai import (
    OpenAI,
    OpenAIEmbedding,
    OpenAISTT,
    OpenAIText,
    OpenAITTS,
)
from hai.providers.providers import (
    ProviderEmbedding,
    ProviderInfo,
    ProviderSTT,
    ProviderText,
    ProviderTTS,
    StreamInterruptedError,
    ToolLoopLimitError,
    TurnResult,
    resolve_api_key,
)

PROVIDERS = ["openai", "anthropic", "google", "ollama", "gptsovits"]

PROVIDER_REGISTRY: Dict[str, Dict[str, Type[Any]]] = {
    "openai": {
        "info": OpenAI,
        "text": OpenAIText,
        "embedding": OpenAIEmbedding,
        "stt": OpenAISTT,
        "tts": OpenAITTS,
    },
    "anthropic": {
        "info": Anthropic,
        "text": AnthropicText,
    },
    "google": {
        "info": Google,
        "text": GoogleText,
        "embedding": GoogleEmbedding,
    },
    "ollama": {
        "info": Ollama,
        "text": OllamaText,
    },
    "gptsovits": {
        "info": GPTSoVITS,
        "tts": GPTSoVITSTTS,
    },
}


def create_text_provider(
    provider: Literal["openai", "anthropic", "google", "ollama"],
    model: str,
    api_key: Optional[str] = None,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> ProviderText:
    """Create a text provider with a unified interface.

    `temperature` and `max_tokens` fall back to each provider's own defaults when None.

    `reasoning_effort` is normalized across providers:
    - OpenAI: forwarded as `reasoning_effort`
    - Anthropic: mapped to `thinking.budget_tokens` (low=1024, medium=4096, high=16000)

    Use `extra_params` to inject provider-specific raw payload fields.
    """
    if provider not in PROVIDER_REGISTRY:
        raise ValueError(f"Unknown provider '{provider}'. Choose from {PROVIDERS}.")
    cls = PROVIDER_REGISTRY[provider].get("text")
    if cls is None:
        raise ValueError(f"Provider '{provider}' has no text implementation.")

    init_kwargs: Dict[str, Any] = {"model": model, "stream": stream, **kwargs}
    if temperature is not None:
        init_kwargs["temperature"] = temperature
    if max_tokens is not None:
        init_kwargs["max_tokens"] = max_tokens
    if api_key is not None:
        init_kwargs["api_key"] = api_key
    if provider in ("openai", "anthropic"):
        init_kwargs["reasoning_effort"] = reasoning_effort
        init_kwargs["extra_params"] = extra_params or {}

    return cls(**init_kwargs)


__all__ = [
    "PROVIDERS",
    "PROVIDER_REGISTRY",
    "create_text_provider",
    "ProviderInfo",
    "ProviderText",
    "ProviderSTT",
    "ProviderTTS",
    "ProviderEmbedding",
    "TurnResult",
    "StreamInterruptedError",
    "ToolLoopLimitError",
    "resolve_api_key",
    "OpenAI",
    "OpenAIText",
    "OpenAIEmbedding",
    "OpenAISTT",
    "OpenAITTS",
    "Anthropic",
    "AnthropicText",
    "Google",
    "GoogleText",
    "GoogleEmbedding",
    "Ollama",
    "OllamaText",
    "GPTSoVITS",
    "GPTSoVITSTTS",
]

from typing import Any, Dict, Literal, Optional, Type

from hai.providers.anthropic import Anthropic, AnthropicText
from hai.providers.google import Google, GoogleEmbedding, GoogleText
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
    TurnResult,
)

PROVIDERS = ["openai", "anthropic", "google", "ollama"]

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
}


def create_text_provider(
    provider: Literal["openai", "anthropic", "google", "ollama"],
    model: str,
    api_key: Optional[str] = None,
    *,
    temperature: float = 0.7,
    max_tokens: int = 1500,
    stream: bool = False,
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> ProviderText:
    """Create a text provider with a unified interface.

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

    init_kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        **kwargs,
    }
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
]

from .providers import ProviderInfo, ProviderText


class Ollama(ProviderInfo):
    def __init__(
        self,
        name: str = "Ollama",
        api_key: str = "your_ollama_api_key",
        base_url: str = "http://localhost:11434/v1",
    ):
        super().__init__(
            name=name,
            api_key=api_key,
            base_url=base_url,
        )


class OllamaText(ProviderText):
    def __init__(
        self,
        name: str = "Ollama",
        api_key: str = "ollama",
        model: str = "gemma3",
        base_url: str = "http://localhost:11434/v1/chat/completions",
        temperature: float = 0.7,
        max_tokens: int = 1500,
        stream: bool = False,
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        """
        Args:
            api_key: Ollama ignores the key, but the OpenAI-compatible API expects one.
            base_url: Full chat completions URL of the Ollama server.
        """
        super().__init__(
            name=name,
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            max_retries=max_retries,
            timeout=timeout,
        )

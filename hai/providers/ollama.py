from .providers import ProviderInfo, ProviderText


class Ollama(ProviderInfo):
    def __init__(self, name: str = "Ollama"):
        super().__init__(
            name=name,
            api_key="your_ollama_api_key",
            base_url="http://localhost:11434/v1",
        )


class OllamaText(ProviderText):
    def __init__(
        self,
        name: str = "Ollama",
        model: str = "gemma3",
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ):
        super().__init__(
            name=name,
            model=model,
            base_url="http://localhost:11434/v1/chat/completions",
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

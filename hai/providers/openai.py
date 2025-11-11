from .providers import ProviderEmbedding, ProviderText


class OpenAI(ProviderText):
    def __init__(
        self,
        name: str = "OpenAI",
        api_key: str = "your_openai_api_key",
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ):
        super().__init__(
            name=name,
            api_key=api_key,
            model=model,
            base_url="https://api.openai.com/v1/chat/completions",
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )


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

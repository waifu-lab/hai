from .providers import ProviderText


class AnthropicText(ProviderText):
    def __init__(
        self,
        name: str = "Anthropic",
        api_key: str = "your_anthropic_api_key",
        model: str = "claude-sonnet-4-5",
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ):
        super().__init__(
            name=name,
            api_key=api_key,
            model=model,
            base_url="https://api.anthropic.com/v1/chat/completions",
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

from typing import Any, Dict, Literal, Optional

from .providers import ProviderInfo, ProviderText


class Anthropic(ProviderInfo):
    def __init__(self, name: str = "Anthropic"):
        super().__init__(
            name=name,
            api_key="your_anthropic_api_key",
            base_url="https://api.anthropic.com/v1",
        )


class AnthropicText(ProviderText):
    def __init__(
        self,
        name: str = "Anthropic",
        api_key: str = "your_anthropic_api_key",
        model: str = "claude-sonnet-4-5",
        temperature: float = 0.7,
        max_tokens: int = 1500,
        stream: bool = False,
        max_retries: int = 3,
        timeout: float = 60.0,
        thinking_mode: Optional[Literal["enabled", "adaptive"]] = None,
        thinking_budget: Optional[int] = None,
    ):
        super().__init__(
            name=name,
            api_key=api_key,
            model=model,
            base_url="https://api.anthropic.com/v1/chat/completions",
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            max_retries=max_retries,
            timeout=timeout,
        )
        self.thinking_mode = thinking_mode
        self.thinking_budget = thinking_budget

    def _extra_payload(self) -> Dict[str, Any]:
        if self.thinking_mode == "adaptive":
            return {"thinking": {"type": "adaptive"}}
        if self.thinking_mode == "enabled" and self.thinking_budget is not None:
            return {"thinking": {"type": "enabled", "budget_tokens": self.thinking_budget}}
        return {}

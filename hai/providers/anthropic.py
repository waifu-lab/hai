from typing import Any, Dict, Literal, Optional

from .providers import ProviderInfo, ProviderText

_EFFORT_TO_BUDGET = {"low": 1024, "medium": 4096, "high": 16000}


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
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        thinking_mode: Optional[Literal["enabled", "adaptive"]] = None,
        thinking_budget: Optional[int] = None,
        extra_params: Optional[Dict[str, Any]] = None,
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
        self.reasoning_effort = reasoning_effort
        self.thinking_mode = thinking_mode
        self.thinking_budget = thinking_budget
        self.extra_params = extra_params or {}

    def _extra_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.thinking_mode == "adaptive":
            payload["thinking"] = {"type": "adaptive"}
        elif self.thinking_mode == "enabled" and self.thinking_budget is not None:
            payload["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
        elif self.reasoning_effort is not None:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": _EFFORT_TO_BUDGET[self.reasoning_effort],
            }
        payload.update(self.extra_params)
        return payload

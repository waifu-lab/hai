from typing import Any, Dict, List, Literal, Optional

from httpx import AsyncClient

from .providers import ProviderInfo, ProviderText, resolve_api_key

_API_KEY_ENV = ("ANTHROPIC_API_KEY",)
_ANTHROPIC_VERSION = "2023-06-01"

_EFFORT_TO_BUDGET = {"low": 1024, "medium": 4096, "high": 16000}
_MIN_THINKING_BUDGET = 1024
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_ANSWER_TOKENS = 1500


def _resolve_thinking(
    thinking_mode: Optional[str],
    thinking_budget: Optional[int],
    reasoning_effort: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Turn the three thinking options into one ``thinking`` payload, rejecting conflicting combinations."""
    if thinking_mode == "adaptive":
        if thinking_budget is not None or reasoning_effort is not None:
            raise ValueError("thinking_mode='adaptive' takes neither thinking_budget nor reasoning_effort.")
        return {"type": "adaptive"}

    if thinking_budget is not None and reasoning_effort is not None:
        raise ValueError("Pass either thinking_budget or reasoning_effort, not both.")
    budget = _EFFORT_TO_BUDGET[reasoning_effort] if reasoning_effort is not None else thinking_budget
    if budget is None:
        if thinking_mode == "enabled":
            raise ValueError("thinking_mode='enabled' needs thinking_budget or reasoning_effort.")
        return None
    if budget < _MIN_THINKING_BUDGET:
        raise ValueError(f"thinking_budget must be >= {_MIN_THINKING_BUDGET}, got {budget}.")
    return {"type": "enabled", "budget_tokens": budget}


class Anthropic(ProviderInfo):
    def __init__(
        self,
        name: str = "Anthropic",
        api_key: Optional[str] = None,
        base_url: str = "https://api.anthropic.com/v1",
    ):
        super().__init__(
            name=name,
            api_key=resolve_api_key(api_key, _API_KEY_ENV, "Anthropic"),
            base_url=base_url,
        )

    async def models(self) -> List[str]:
        """Model ids from Anthropic's native ``/models`` endpoint. Follows every result page."""
        headers = {"x-api-key": self.api_key, "anthropic-version": _ANTHROPIC_VERSION}
        params: Dict[str, Any] = {"limit": 1000}
        ids: List[str] = []
        async with AsyncClient(timeout=30.0) as client:
            while True:
                response = await client.get(f"{self.base_url}/models", headers=headers, params=params)
                response.raise_for_status()
                res = response.json()
                ids.extend(item["id"] for item in res.get("data", []))
                if not res.get("has_more") or not res.get("last_id"):
                    return ids
                params["after_id"] = res["last_id"]


class AnthropicText(ProviderText):
    """Claude through Anthropic's OpenAI-compatible endpoint.

    Extended thinking is off by default. Turn it on with one of:

    * ``reasoning_effort``: budget shortcut, low=1024, medium=4096, high=16000 tokens.
    * ``thinking_budget``: explicit budget in tokens (>= 1024). ``thinking_mode="enabled"``
      is implied and may be omitted.
    * ``thinking_mode="adaptive"``: the model decides how much to think.

    Conflicting combinations raise ``ValueError`` instead of being ignored.

    With thinking on, Anthropic only accepts the default temperature, so ``temperature``
    is left unset (passing anything other than 1 raises). Thinking tokens count against
    ``max_tokens``, so when it is not given it defaults to the budget plus 1500 tokens
    for the answer (adaptive uses the "high" budget). Without thinking the defaults are
    ``temperature=0.7`` and ``max_tokens=1500``.
    """

    def __init__(
        self,
        name: str = "Anthropic",
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        max_retries: int = 3,
        timeout: float = 60.0,
        max_tool_rounds: int = 10,
        base_url: str = "https://api.anthropic.com/v1/chat/completions",
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        thinking_mode: Optional[Literal["enabled", "adaptive"]] = None,
        thinking_budget: Optional[int] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ):
        thinking = _resolve_thinking(thinking_mode, thinking_budget, reasoning_effort)
        if thinking is None:
            temperature = _DEFAULT_TEMPERATURE if temperature is None else temperature
            max_tokens = _DEFAULT_ANSWER_TOKENS if max_tokens is None else max_tokens
        else:
            if temperature is not None and temperature != 1:
                raise ValueError(f"Extended thinking only supports temperature=1, got {temperature}.")
            budget = thinking.get("budget_tokens", _EFFORT_TO_BUDGET["high"])
            if max_tokens is None:
                max_tokens = budget + _DEFAULT_ANSWER_TOKENS
            elif "budget_tokens" in thinking and max_tokens <= budget:
                raise ValueError(f"max_tokens ({max_tokens}) must be greater than the thinking budget ({budget}).")

        super().__init__(
            name=name,
            api_key=resolve_api_key(api_key, _API_KEY_ENV, "Anthropic"),
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            max_retries=max_retries,
            timeout=timeout,
            max_tool_rounds=max_tool_rounds,
        )
        self.reasoning_effort = reasoning_effort
        self.thinking_mode = thinking_mode
        self.thinking_budget = thinking_budget
        self.thinking = thinking
        self.extra_params = extra_params or {}

    def _extra_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.thinking is not None:
            payload["thinking"] = self.thinking
        payload.update(self.extra_params)
        return payload

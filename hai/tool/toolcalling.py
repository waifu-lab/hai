import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union


@dataclass
class ImageResult:
    data: bytes
    media_type: str = "image/png"
    description: str = ""


class Tool:
    """
    Wraps a Python callable as an OpenAI-compatible function tool.

    Example
    ```python
    def get_weather(city: str) -> str:
        return f"Sunny in {city}"

    tool = Tool(
        name="get_weather",
        description="Get the weather for a city",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"],
        },
        func=get_weather,
    )
    ```
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    def to_dict(self) -> Dict[str, Any]:
        """Return the OpenAI tool schema dict."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def call(self, arguments: str) -> Union[str, ImageResult]:
        """Parse arguments JSON string and invoke the wrapped function.

        Returns an error string instead of raising if the function fails,
        so that tool execution errors do not abort the LLM conversation loop.
        ImageResult is returned as-is so the caller can inject it as a proper
        multimodal user message instead of stringifying the bytes.
        """
        try:
            kwargs = json.loads(arguments)
        except json.JSONDecodeError:
            kwargs = {}

        try:
            if inspect.iscoroutinefunction(self.func):
                result = await self.func(**kwargs)
            else:
                result = self.func(**kwargs)
        except Exception as e:
            return f"Tool error ({self.name}): {e}"

        if isinstance(result, ImageResult):
            return result
        return str(result)


def tool(
    name: Optional[str] = None,
    description: str = "",
    parameters: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable], Tool]:
    """
    Decorator to convert a function into a Tool instance.

    Example:
    ```python
    @tool(description="Add two numbers", parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
    })
    def add(a: float, b: float) -> float:
        return a + b
    ```
    """

    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").strip()
        tool_params = parameters or {"type": "object", "properties": {}, "required": []}
        return Tool(name=tool_name, description=tool_desc, parameters=tool_params, func=func)

    return decorator

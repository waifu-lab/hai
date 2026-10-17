import base64
import collections.abc
import functools
import inspect
import json
import logging
import re
import types
from dataclasses import dataclass
from enum import Enum
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}
_ARRAY_ORIGINS = (list, tuple, set, frozenset, collections.abc.Sequence, collections.abc.Set)
_OBJECT_ORIGINS = (dict, collections.abc.Mapping)
_UNION_TYPE = getattr(types, "UnionType", None)
_SECTION = re.compile(r"^(\w[\w ]*):\s*$")
_PARAM = re.compile(r"^\*{0,2}(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")


class ToolArgumentsError(ValueError):
    """The LLM's tool-call arguments are not a valid JSON object."""


@dataclass
class ImageResult:
    data: bytes
    media_type: str = "image/png"
    description: str = ""

    def to_data_url(self) -> str:
        return f"data:{self.media_type};base64,{base64.b64encode(self.data).decode('ascii')}"


def _type_to_schema(tp: Any) -> Dict[str, Any]:
    """Map a Python type hint to a JSON Schema fragment. Unknown types map to ``{}`` (any)."""
    origin = get_origin(tp)
    args = get_args(tp)

    if origin is Annotated:
        schema = _type_to_schema(args[0])
        description = next((a for a in args[1:] if isinstance(a, str)), None)
        if description:
            schema["description"] = description
        return schema
    if origin is Literal:
        schema: Dict[str, Any] = {"enum": list(args)}
        value_types = {type(a) for a in args}
        if len(value_types) == 1 and next(iter(value_types)) in _JSON_TYPES:
            schema["type"] = _JSON_TYPES[next(iter(value_types))]
        return schema
    if origin is Union or (_UNION_TYPE is not None and isinstance(tp, _UNION_TYPE)):
        options = [a for a in args if a is not type(None)]
        if len(options) == 1:
            return _type_to_schema(options[0])
        return {"anyOf": [_type_to_schema(a) for a in options]}
    if tp in _JSON_TYPES:
        return {"type": _JSON_TYPES[tp]}
    if tp in _ARRAY_ORIGINS or origin in _ARRAY_ORIGINS:
        schema = {"type": "array"}
        if args and args[0] is not Ellipsis:
            schema["items"] = _type_to_schema(args[0])
        return schema
    if tp in _OBJECT_ORIGINS or origin in _OBJECT_ORIGINS:
        return {"type": "object"}
    if inspect.isclass(tp) and issubclass(tp, Enum):
        return _type_to_schema(Literal[tuple(m.value for m in tp)])
    if inspect.isclass(tp) and issubclass(tp, BaseModel):
        return tp.model_json_schema()
    return {}


def _parse_docstring(doc: str) -> Tuple[str, Dict[str, str]]:
    """Split a Google-style docstring into ``(summary, {param: description})``.

    The summary is everything before the first section header (``Args:``,
    ``Returns:`` ...). Parameter descriptions come from the ``Args:`` section.
    """
    summary: List[str] = []
    params: Dict[str, str] = {}
    section: Optional[str] = None
    current: Optional[str] = None
    param_indent: Optional[int] = None

    for line in inspect.cleandoc(doc).splitlines():
        header = _SECTION.match(line)
        if header:
            section = header.group(1).lower()
            current, param_indent = None, None
            continue
        if section is None:
            summary.append(line)
            continue
        if section not in ("args", "arguments", "parameters") or not line.strip():
            continue

        indent = len(line) - len(line.lstrip())
        if param_indent is None:
            param_indent = indent
        match = _PARAM.match(line.strip())
        if match and indent == param_indent:
            current = match.group(1)
            params[current] = match.group(2).strip()
        elif current:
            params[current] = f"{params[current]} {line.strip()}".strip()

    return "\n".join(summary).strip(), params


def _infer_parameters(func: Callable[..., Any], param_docs: Dict[str, str]) -> Dict[str, Any]:
    """Build the tool's JSON Schema from *func*'s signature and type hints.

    Parameters without a default are required. ``Optional[X]`` maps to ``X``,
    ``Literal`` and ``Enum`` map to ``enum``, and ``Annotated[X, "text"]``
    sets the parameter description (overriding the docstring).
    """
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception as e:
        raise TypeError(
            f"Cannot resolve type hints of tool function '{func.__name__}': {e}. Pass `parameters=` explicitly."
        ) from e

    properties: Dict[str, Any] = {}
    required: List[str] = []
    for param in inspect.signature(func).parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        schema = _type_to_schema(hints.get(param.name, Any))
        if param.name in param_docs and "description" not in schema:
            schema["description"] = param_docs[param.name]
        properties[param.name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(param.name)

    return {"type": "object", "properties": properties, "required": required}


class Tool:
    """
    Wraps a Python callable as an OpenAI-compatible function tool.

    Usually created with the :func:`tool` decorator. The Tool stays callable
    like the original function.

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
        func: Callable[..., Any],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func
        functools.update_wrapper(self, func, updated=())

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Tool(name='{self.name}')"

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

    async def invoke(self, arguments: str) -> Union[str, ImageResult]:
        """Parse the LLM's JSON arguments and invoke the wrapped function, raising on failure.

        Raises :class:`ToolArgumentsError` for arguments that are not a JSON object,
        and lets exceptions from the function propagate. For an agent loop that
        reports tool errors itself; use :meth:`call` to get an error string instead.
        """
        try:
            kwargs = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as e:
            raise ToolArgumentsError(f"arguments are not valid JSON: {e}") from e
        if not isinstance(kwargs, dict):
            raise ToolArgumentsError(f"arguments must be a JSON object, got {type(kwargs).__name__}")

        result = self.func(**kwargs)
        if inspect.isawaitable(result):
            result = await result

        if isinstance(result, ImageResult):
            return result
        return str(result)

    async def call(self, arguments: str) -> Union[str, ImageResult]:
        """Like :meth:`invoke`, but failures are logged and returned as an error string.

        Failures (invalid JSON, wrong arguments, exceptions raised by the
        function) do not raise, so the LLM can see what went wrong and the
        conversation loop keeps going. ImageResult is returned as-is;
        :func:`run_tool_calls` turns it into a multimodal message.
        """
        try:
            return await self.invoke(arguments)
        except ToolArgumentsError as e:
            return f"Tool error ({self.name}): {e}"
        except Exception as e:
            logger.exception("tool %s failed", self.name)
            return f"Tool error ({self.name}): {e}"


@overload
def tool(name: Callable[..., Any]) -> Tool: ...


@overload
def tool(
    name: Optional[str] = None,
    description: str = "",
    parameters: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    name: Union[str, Callable[..., Any], None] = None,
    description: str = "",
    parameters: Optional[Dict[str, Any]] = None,
) -> Union[Tool, Callable[[Callable[..., Any]], Tool]]:
    """
    Decorator to convert a function into a Tool instance.

    Works with or without parentheses. By default the name is the function name,
    the description is the docstring summary, and the parameter schema is
    inferred from the signature, type hints and the docstring's ``Args:`` section.

    Example:
    ```python
    @tool
    def add(a: float, b: float = 0) -> float:
        \"\"\"Add two numbers.

        Args:
            a: First number.
            b: Second number.
        \"\"\"
        return a + b

    add(1, 2)  # still callable -> 3

    @tool(name="search", description="Search the docs")
    async def search_docs(query: str, mode: Literal["fast", "deep"] = "fast") -> str:
        ...
    ```
    """
    if callable(name):
        return tool()(name)

    def decorator(func: Callable[..., Any]) -> Tool:
        summary, param_docs = _parse_docstring(func.__doc__ or "")
        return Tool(
            name=name or func.__name__,
            description=description or summary,
            parameters=parameters or _infer_parameters(func, param_docs),
            func=func,
        )

    return decorator


async def run_tool_calls(tool_calls: Sequence[Dict[str, Any]], tools: Sequence[Tool]) -> List[Dict[str, Any]]:
    """Execute OpenAI-format tool calls and return the messages to append to the conversation.

    One ``tool`` message is returned per call, in order, so every ``tool_call_id``
    is answered right after the assistant turn. A tool message can only carry
    text, so each ImageResult is answered with a short text and the image itself
    follows as a ``user`` message with an ``image_url`` part, after all tool messages.
    """
    tool_map = {t.name: t for t in tools}
    tool_messages: List[Dict[str, Any]] = []
    image_messages: List[Dict[str, Any]] = []

    for tool_call in tool_calls:
        func_name = tool_call["function"]["name"]
        tool_id = tool_call["id"]
        target = tool_map.get(func_name)
        result = await target.call(tool_call["function"]["arguments"]) if target else f"Tool '{func_name}' not found"

        if isinstance(result, ImageResult):
            note = f"Image returned by tool '{func_name}' (call {tool_id}). {result.description}".strip()
            image_messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": note},
                        {"type": "image_url", "image_url": {"url": result.to_data_url()}},
                    ],
                }
            )
            content = "The image is attached in the next user message."
        else:
            content = result

        tool_messages.append({"role": "tool", "tool_call_id": tool_id, "content": content})

    return tool_messages + image_messages

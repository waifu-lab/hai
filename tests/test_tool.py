import asyncio
from typing import List, Literal, Optional

from hai.tool import run_tool_calls, tool


@tool
def search(query: str, mode: Literal["fast", "deep"] = "fast", tags: Optional[List[str]] = None) -> str:
    """Search the docs.

    Args:
        query: What to look for.
        mode: Search depth.
    """
    return f"{query}:{mode}:{tags}"


def test_schema_inferred_from_signature_and_docstring():
    fn = search.to_dict()["function"]
    assert fn["description"] == "Search the docs."
    props = fn["parameters"]["properties"]
    assert props["query"] == {"type": "string", "description": "What to look for."}
    assert props["mode"]["enum"] == ["fast", "deep"]
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert fn["parameters"]["required"] == ["query"]


def test_run_tool_calls_reports_errors_instead_of_raising():
    calls = [
        {"id": "1", "function": {"name": "search", "arguments": '{"query": "x"}'}},
        {"id": "2", "function": {"name": "search", "arguments": "not json"}},
        {"id": "3", "function": {"name": "missing", "arguments": "{}"}},
    ]
    messages = asyncio.run(run_tool_calls(calls, [search]))
    assert [m["tool_call_id"] for m in messages] == ["1", "2", "3"]
    assert messages[0]["content"] == "x:fast:None"
    assert messages[1]["content"].startswith("Tool error (search)")
    assert "not found" in messages[2]["content"]

import asyncio

import pytest

from hai.core import BaseNode, Flow


class Record(BaseNode):
    """Appends its name to ``context["seen"]`` and returns *result* (the context by default)."""

    def __init__(self, name, result=None):
        super().__init__(name)
        self.result = result

    async def exec(self, context):
        context["seen"].append(self.name)
        return context if self.result is None else self.result


def run(start, **context):
    return asyncio.run(Flow(start).run({"seen": [], **context}))["seen"]


def test_chain_runs_in_order():
    a, b, c = Record("a"), Record("b"), Record("c")
    a >> b >> c
    assert run(a) == ["a", "b", "c"]


def test_branch_on_returned_string():
    review, publish, rewrite = Record("review", "revise"), Record("publish"), Record("rewrite")
    review >> {"approved": publish, "revise": rewrite}
    assert run(review) == ["review", "rewrite"]


def test_branch_on_returned_next_dict():
    review, publish, rewrite = Record("review", {"next": "approved"}), Record("publish"), Record("rewrite")
    review >> {"approved": publish, "revise": rewrite}
    assert run(review) == ["review", "publish"]


def test_next_key_in_context_does_not_pick_a_branch():
    a, b, x = Record("a"), Record("b"), Record("x")
    a >> b
    a >> {"special": x}
    assert run(a, next="special") == ["a", "b"]


def test_run_without_context_starts_empty():
    class Write(BaseNode):
        async def exec(self, context):
            context["count"] = context.get("count", 0) + 1

    flow = Flow(Write("write"))
    assert asyncio.run(flow.run()) == {"count": 1}
    # A fresh dict every run, not a shared default
    assert asyncio.run(flow.run()) == {"count": 1}


def test_run_returns_the_given_context():
    given = {"seen": []}
    assert asyncio.run(Flow(Record("a")).run(given)) is given


def test_flow_from_last_node_warns():
    a, b, c = Record("a"), Record("b"), Record("c")
    with pytest.warns(UserWarning, match="last node of a chain"):
        Flow(a >> b >> c)


def test_flow_from_first_node_does_not_warn(recwarn):
    a, b = Record("a"), Record("b")
    a >> b
    Flow(a)
    assert not recwarn.list


def test_on_error_false_skips_failed_node():
    class Boom(BaseNode):
        async def exec(self, _context):
            raise RuntimeError("boom")

    a, boom, c = Record("a"), Boom("boom"), Record("c")
    a >> boom >> c
    errors = []
    flow = Flow(a).on_error(lambda node, exc: errors.append((node.name, str(exc))) or False)
    context = asyncio.run(flow.run({"seen": []}))
    assert context["seen"] == ["a", "c"]
    assert errors == [("boom", "boom")]

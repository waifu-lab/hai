from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, Optional

from .node import BaseNode

logger = logging.getLogger(__name__)


class Flow:
    """Runs a pre-chained node pipeline with execution control and error handling.

    ```python
    query_embed >> retriever >> context_builder

    flow = Flow(query_embed)
    context = await flow.run({"query": "..."})
    ```
    """

    def __init__(self, start: BaseNode):
        self.start = start
        self._timeout: Optional[float] = None
        self._on_error: Optional[Callable[[BaseNode, Exception], bool]] = None

    def timeout(self, seconds: float) -> "Flow":
        """Set total execution timeout in seconds."""
        self._timeout = seconds
        return self

    def on_error(self, handler: Callable[[BaseNode, Exception], bool]) -> "Flow":
        """Handle node errors.

        Return True to re-raise and stop the pipeline, False to skip the failed
        node and continue with its default next node.
        """
        self._on_error = handler
        return self

    async def _execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        current: Optional[BaseNode] = self.start
        while current:
            node = current
            try:
                current = await node.execute(context)
                logger.debug("node %s completed", node.name)
            except Exception as exc:
                logger.error("node %s failed: %s", node.name, exc)
                if self._on_error is None or self._on_error(node, exc):
                    raise
                current = node.next(None)
        return context

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self._timeout:
            return await asyncio.wait_for(self._execute(context), timeout=self._timeout)
        return await self._execute(context)

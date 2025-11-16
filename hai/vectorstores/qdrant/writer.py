from typing import Any, Dict

from hai.core import BaseNode
from hai.vectorstores.qdrant.QdrantClient import QdrantStore


class QdrantStoreWriter(BaseNode):
    def __init__(
        self,
        name: str = "VectorStoreWriter",
        store: QdrantStore = None,
    ):
        super().__init__(name)
        self.store = store

    async def exec(self, context: Dict[str, Any]) -> Dict[str, Any]:
        chunks = context.get("embedded_chunks", [])

        if not chunks:
            context["stored_ids"] = []
            return context

        ids = self.store.write(chunks)
        context["stored_ids"] = ids
        context["stored_count"] = len(ids)

        return context

from typing import Any, Dict

from hai.core import BaseNode
from hai.vectorstores.qdrant.QdrantClient import QdrantStore


class QdrantStoreWriter(BaseNode):
    """Upsert embedded chunks into *store*.

    Context:
        reads: ``embedded_chunks`` (List[dict] from EmbeddingProcessor)
        writes: ``stored_ids`` (List[str]), ``stored_count`` (int)
    """

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
            context["stored_count"] = 0
            return context

        ids = await self.store.write(chunks)
        context["stored_ids"] = ids
        context["stored_count"] = len(ids)

        return context

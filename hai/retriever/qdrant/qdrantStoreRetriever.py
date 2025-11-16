from typing import Any, Dict, Optional

from hai.core import BaseNode
from hai.vectorstores.qdrant.QdrantClient import QdrantStore


class VectorStoreRetriever(BaseNode):
    def __init__(
        self,
        name: str = "VectorStoreRetriever",
        store: QdrantStore = None,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ):
        super().__init__(name)
        self.store = store
        self.top_k = top_k
        self.score_threshold = score_threshold

    async def exec(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query_embedding = context.get("query_embedding", [])

        if not query_embedding:
            raise ValueError("context 中必須包含 'query_embedding'")

        results = self.store.query(
            query_embedding=query_embedding,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )

        context["retrieved_documents"] = results
        context["retrieved_count"] = len(results)

        return context

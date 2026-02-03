from typing import Any, Dict

from hai.core import BaseNode


class EmbeddingProcessor(BaseNode):
    def __init__(
        self,
        name: str = "EmbeddingProcessor",
        embedder: BaseNode = None,
        batch_size: int = 32,
    ):
        super().__init__(name)
        self.embedder = embedder
        self.batch_size = batch_size

    async def exec(self, context: Dict[str, Any]) -> Dict[str, Any]:
        chunks = context.get("chunks", [])

        if not chunks:
            context["embedded_chunks"] = []
            return context

        texts = [chunk["content"] for chunk in chunks]

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embed_context = {"input": batch}
            await self.embedder.exec(embed_context)
            batch_embeddings = embed_context.get("embeddings", [])
            all_embeddings.extend(batch_embeddings)

        for i, chunk in enumerate(chunks):
            chunk["embedding"] = all_embeddings[i]

        context["embedded_chunks"] = chunks

        return context


class QueryEmbedder(BaseNode):
    def __init__(
        self,
        name: str = "QueryEmbedder",
        embedder: BaseNode = None,
    ):
        super().__init__(name)
        self.embedder = embedder

    async def exec(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query", "")

        if not query:
            raise ValueError("context 中必須包含 'query'")

        embed_context = {"input": query}
        await self.embedder.exec(embed_context)
        context["query_embedding"] = embed_context.get("embedding", [])

        return context

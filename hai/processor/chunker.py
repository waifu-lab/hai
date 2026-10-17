from typing import Any, Dict, Optional

from hai.core import BaseNode


class ChunkProcessor(BaseNode):
    """Split every document with *splitter* and flatten the pieces into chunk dicts.

    Each chunk dict holds ``content``, ``name``, ``chunk_index``, ``total_chunks``
    and the document's metadata. Without a splitter each document becomes one chunk.

    Context:
        reads: ``documents`` (List[Document])
        writes: ``chunks`` (List[dict]), ``total_chunks`` (int)
    """

    def __init__(
        self,
        name: str = "ChunkProcessor",
        splitter: Optional[BaseNode] = None,
    ):
        super().__init__(name)
        self.splitter = splitter

    async def exec(self, context: Dict[str, Any]) -> Dict[str, Any]:
        documents = context.get("documents", [])
        all_chunks = []

        for doc in documents:
            doc_name = doc.get("name", "unknown")
            doc_content = doc.get("content", "")
            doc_metadata = {
                k: v for k, v in doc.items() if k not in ["content", "name"]
            }

            if self.splitter:
                splitter_context = {"text": doc_content}
                await self.splitter.exec(splitter_context)
                chunks = splitter_context.get("text_chunks", [doc_content])
            else:
                chunks = [doc_content]

            for i, chunk in enumerate(chunks):
                chunk_data = {
                    "content": chunk,
                    "name": doc_name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **doc_metadata,
                }
                all_chunks.append(chunk_data)

        context["chunks"] = all_chunks
        context["total_chunks"] = len(all_chunks)

        return context

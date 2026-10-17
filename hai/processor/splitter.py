from typing import Any, Dict, List

from hai.core import BaseNode


class Splitter(BaseNode):
    """Split text on a single separator and merge pieces up to *chunk_size* characters.

    Context:
        reads: ``text`` (str)
        writes: ``text_chunks`` (List[str]), ``chunk_count`` (int)
    """

    def __init__(
        self,
        name: str = "Splitter",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n",
        keep_separator: bool = True,
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap must be >= 0 and < chunk_size ({chunk_size}), got {chunk_overlap}")
        super().__init__(name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.keep_separator = keep_separator

    async def exec(self, context: Dict[str, Any]) -> Dict[str, Any]:
        text = context.get("text", "")

        if not text:
            context["text_chunks"] = []
            return context

        chunks = self._split_text(text)
        context["text_chunks"] = chunks
        context["chunk_count"] = len(chunks)

        return context

    def _split_text(self, text: str) -> List[str]:
        if self.separator:
            splits = text.split(self.separator)
            if self.keep_separator:
                splits = [f"{s}{self.separator}" if i < len(splits) - 1 else s for i, s in enumerate(splits)]
        else:
            splits = list(text)

        return self._merge_splits(splits)

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Greedily pack splits into chunks of at most *chunk_size* characters.

        When a chunk is emitted, its trailing splits (at most *chunk_overlap*
        characters in total) are carried over to start the next chunk. A single
        split longer than *chunk_size* becomes its own oversized chunk.
        """
        chunks = []
        current_chunk: List[str] = []
        current_length = 0

        for split in splits:
            split_length = len(split)

            if current_length + split_length > self.chunk_size and current_chunk:
                chunks.append("".join(current_chunk))

                while current_chunk and (
                    current_length > self.chunk_overlap or current_length + split_length > self.chunk_size
                ):
                    current_length -= len(current_chunk.pop(0))

            current_chunk.append(split)
            current_length += split_length

        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks

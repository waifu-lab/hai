import re
from typing import Any, Dict, List, Optional

from core.node import BaseNode


class RecursiveSplitter(BaseNode):
    def __init__(
        self,
        name: str = "RecursiveSplitter",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
    ):
        super().__init__(name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.keep_separator = keep_separator

        if separators is None:
            self.separators = [
                "\n\n",
                "\n",
                "。",
                "！",
                "？",
                ". ",
                "! ",
                "? ",
                " ",
                "",
            ]
        else:
            self.separators = separators

    async def exec(self, context: Dict[str, Any]) -> Dict[str, Any]:
        text = context.get("text", "")

        if not text:
            context["chunks"] = []
            return context

        chunks = self._split_text(text, self.separators)
        context["chunks"] = chunks
        context["chunk_count"] = len(chunks)

        return context

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []

        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break

            if re.search(re.escape(sep), text):
                separator = sep
                new_separators = separators[i + 1 :]
                break

        splits = self._split_by_separator(text, separator)

        good_splits = []
        for split in splits:
            if len(split) < self.chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits)
                    final_chunks.extend(merged)
                    good_splits = []

                if not new_separators:
                    final_chunks.append(split)
                else:
                    other_chunks = self._split_text(split, new_separators)
                    final_chunks.extend(other_chunks)

        if good_splits:
            merged = self._merge_splits(good_splits)
            final_chunks.extend(merged)

        return final_chunks

    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        if separator:
            if self.keep_separator:
                splits = re.split(f"({re.escape(separator)})", text)
                splits = [
                    splits[i] + (splits[i + 1] if i + 1 < len(splits) else "")
                    for i in range(0, len(splits), 2)
                ]
            else:
                splits = text.split(separator)
            return [s for s in splits if s]
        else:
            return list(text)

    def _merge_splits(self, splits: List[str]) -> List[str]:
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = len(split)

            if current_length + split_length > self.chunk_size and current_chunk:
                chunks.append("".join(current_chunk))

                overlap_size = 0
                overlap_length = 0
                for i in range(len(current_chunk) - 1, -1, -1):
                    overlap_length += len(current_chunk[i])
                    if overlap_length >= self.chunk_overlap:
                        overlap_size = len(current_chunk) - i
                        break

                if overlap_size > 0:
                    current_chunk = current_chunk[-overlap_size:]
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(split)
            current_length += split_length

        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks

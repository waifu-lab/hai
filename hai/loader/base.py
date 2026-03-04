from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from hai.core import BaseNode, Document

_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "gbk"]


def detect_encoding(file_path: Path, forced: Optional[str] = None) -> str:
    """Return the first encoding that successfully decodes the file head."""
    if forced:
        return forced
    for enc in _ENCODINGS:
        try:
            with open(file_path, "r", encoding=enc) as f:
                f.read(4096)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    return "latin-1"


class BaseLoader(BaseNode):
    def __init__(self, name: str):
        super().__init__(name)
        self.documents: List[Document] = []

    def _gather_files(self, source: Path, pattern: str, recursive: bool) -> List[Path]:
        """Return all matching files under *source* (or *source* itself if a file)."""
        if source.is_file():
            return [source]
        if source.is_dir():
            glob_fn = source.rglob if recursive else source.glob
            return [f for f in glob_fn(pattern) if f.is_file()]
        raise FileNotFoundError(f"Source not found: {source}")

    @abstractmethod
    async def load(self) -> List[Document]:
        pass

    async def exec(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.documents = await self.load()
        context["documents"] = self.documents
        context["document_count"] = len(self.documents)
        return context

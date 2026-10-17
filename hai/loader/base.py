import asyncio
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, TypeVar, Union

from hai.core import BaseNode, Document

logger = logging.getLogger(__name__)

_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "gbk"]

T = TypeVar("T")


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
    """Base class for loaders. Subclasses implement :meth:`load`.

    A source that fails to load (unreadable file, bad JSON, HTTP error ...) is
    logged and skipped, so one bad file does not abort a whole directory. Pass
    ``strict=True`` to raise on the first failure instead. Failed sources never
    produce empty Documents.

    Context:
        writes: ``documents`` (List[Document]), ``document_count`` (int)
    """

    def __init__(self, name: str, strict: bool = False):
        super().__init__(name)
        self.strict = strict
        self.documents: List[Document] = []

    def _gather_files(self, source: Path, pattern: str, recursive: bool) -> List[Path]:
        """Return all matching files under *source* (or *source* itself if a file)."""
        if source.is_file():
            return [source]
        if source.is_dir():
            glob_fn = source.rglob if recursive else source.glob
            return [f for f in glob_fn(pattern) if f.is_file()]
        raise FileNotFoundError(f"Source not found: {source}")

    async def _load_each(
        self,
        sources: Sequence[T],
        load_one: Callable[[T], Awaitable[Union[Document, List[Document]]]],
    ) -> List[Document]:
        """Run *load_one* on every source concurrently and flatten the results, applying the error policy."""
        results = await asyncio.gather(*(load_one(source) for source in sources), return_exceptions=True)
        documents: List[Document] = []
        for source, result in zip(sources, results):
            if isinstance(result, BaseException):
                if self.strict or not isinstance(result, Exception):
                    raise result
                logger.warning("%s: failed to load %s: %s", self.name, source, result)
                continue
            documents.extend(result if isinstance(result, list) else [result])
        return documents

    @abstractmethod
    async def load(self) -> List[Document]:
        pass

    async def exec(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.documents = await self.load()
        context["documents"] = self.documents
        context["document_count"] = len(self.documents)
        return context

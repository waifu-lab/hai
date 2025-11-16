import asyncio
from pathlib import Path
from typing import List, Optional, Union

from hai.core import Document

from .base import BaseLoader


class TextLoader(BaseLoader):
    """
    Load plain text files.

    Supports:
    - Single file or directory of files
    - Automatic encoding detection (tries UTF-8, fallback to latin-1)
    - Glob patterns (*.txt, **/*.txt)
    - Custom file extensions

    ```python
    loader = TextLoader("text_loader", source="./docs/readme.txt")
    loader = TextLoader("text_loader", source="./docs", pattern="*.txt")
    loader = TextLoader("text_loader", source="./data", pattern="**/*.log", recursive=True)
    ```
    """  # noqa: E501

    def __init__(
        self,
        name: str,
        source: Union[str, Path],
        pattern: str = "*.txt",
        encoding: Optional[str] = None,
        recursive: bool = False,
    ):
        """
        Args:
            name: Node name
            source: File path or directory path
            pattern: Glob pattern for matching files (default: *.txt)
            encoding: Force specific encoding (default: auto-detect)
            recursive: Search subdirectories (default: False)
        """
        super().__init__(name)
        self.source = Path(source)
        self.pattern = pattern
        self.encoding = encoding
        self.recursive = recursive

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding by trying common encodings."""
        if self.encoding:
            return self.encoding

        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "gbk"]

        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    f.read()
                return enc
            except (UnicodeDecodeError, LookupError):
                continue

        return "latin-1"

    async def _load_file(self, file_path: Path) -> Document:
        """Load a single text file."""
        encoding = self._detect_encoding(file_path)

        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(
            None, lambda: file_path.read_text(encoding=encoding)
        )

        metadata = {
            "source": str(file_path),
            "encoding": encoding,
            "size": file_path.stat().st_size,
            "extension": file_path.suffix,
        }

        return Document(content=content, metadata=metadata, name=file_path.name)

    async def load(self) -> List[Document]:
        """Load text file(s) from source."""
        documents = []

        if self.source.is_file():
            doc = await self._load_file(self.source)
            documents.append(doc)

        elif self.source.is_dir():
            if self.recursive:
                files = list(self.source.rglob(self.pattern))
            else:
                files = list(self.source.glob(self.pattern))

            tasks = [self._load_file(f) for f in files if f.is_file()]
            documents = await asyncio.gather(*tasks)

        else:
            raise FileNotFoundError(f"Source not found: {self.source}")

        return documents

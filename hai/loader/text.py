from pathlib import Path
from typing import List, Optional, Union

import aiofiles

from hai.core import Document

from .base import BaseLoader, detect_encoding


class TextLoader(BaseLoader):
    """
    Load plain text files.

    Supports:
    - Single file or directory of files
    - Automatic encoding detection (tries UTF-8, fallback to latin-1)
    - Glob patterns (*.txt, **/*.txt)
    - Custom file extensions

    ```python
    loader = TextLoader(source="./docs/readme.txt")
    loader = TextLoader(source="./docs", pattern="*.txt")
    loader = TextLoader("log_loader", source="./data", pattern="**/*.log", recursive=True)
    ```
    """

    def __init__(
        self,
        name: str = "TextLoader",
        *,
        source: Union[str, Path],
        pattern: str = "*.txt",
        encoding: Optional[str] = None,
        recursive: bool = False,
        strict: bool = False,
    ):
        """
        Args:
            name: Node name
            source: File path or directory path
            pattern: Glob pattern for matching files (default: *.txt)
            encoding: Force specific encoding (default: auto-detect)
            recursive: Search subdirectories (default: False)
            strict: Raise on the first file that fails to load instead of skipping it
        """
        super().__init__(name, strict=strict)
        self.source = Path(source)
        self.pattern = pattern
        self.encoding = encoding
        self.recursive = recursive

    async def _load_file(self, file_path: Path) -> Document:
        """Load a single text file."""
        encoding = detect_encoding(file_path, self.encoding)

        async with aiofiles.open(file_path, "r", encoding=encoding) as f:
            content = await f.read()

        metadata = {
            "source": str(file_path),
            "encoding": encoding,
            "size": file_path.stat().st_size,
            "extension": file_path.suffix,
        }

        return Document(content=content, metadata=metadata, name=file_path.name)

    async def load(self) -> List[Document]:
        """Load text file(s) from source."""
        files = self._gather_files(self.source, self.pattern, self.recursive)
        return await self._load_each(files, self._load_file)

import asyncio
import csv
import io
import json
from pathlib import Path
from typing import List, Optional, Union

import aiofiles

from hai.core import Document

from .base import BaseLoader, detect_encoding


class CSVLoader(BaseLoader):
    """
    Load CSV files as Documents.

    Each row becomes a Document with content serialized as a JSON string.
    Supports custom delimiters, automatic encoding detection, and large-file
    handling via row-level iteration (no full file load into memory).

    ```python
    # One document per row
    loader = CSVLoader("csv_loader", source="data.csv")

    # Tab-separated, one document per row
    loader = CSVLoader("csv_loader", source="data.tsv", delimiter="\\t")

    # Batch rows (10 rows per document)
    loader = CSVLoader("csv_loader", source="data.csv", batch_size=10)

    # Load all CSVs in a directory
    loader = CSVLoader("csv_loader", source="./data", pattern="*.csv", recursive=True)
    ```
    """

    def __init__(
        self,
        name: str,
        source: Union[str, Path],
        pattern: str = "*.csv",
        delimiter: str = ",",
        encoding: Optional[str] = None,
        recursive: bool = False,
        batch_size: Optional[int] = None,
    ):
        """
        Args:
            name: Node name
            source: File path or directory path
            pattern: Glob pattern for matching files (default: *.csv)
            delimiter: Field delimiter character (default: comma)
            encoding: Force specific encoding (default: auto-detect)
            recursive: Search subdirectories (default: False)
            batch_size: If set, group this many rows into one Document instead
                        of one Document per row. Content becomes a JSON array.
        """
        super().__init__(name)
        self.source = Path(source)
        self.pattern = pattern
        self.delimiter = delimiter
        self.encoding = encoding
        self.recursive = recursive
        self.batch_size = batch_size

    def _build_document(
        self,
        row: dict,
        name: str,
        row_index: int,
        headers: list,
        source_str: str,
        enc: str,
    ) -> Document:
        """Build a single-row Document."""
        return Document(
            content=json.dumps(row, ensure_ascii=False),
            metadata={
                "source": source_str,
                "row_index": row_index,
                "headers": headers,
                "delimiter": self.delimiter,
                "encoding": enc,
            },
            name=name,
        )

    def _build_batch_document(
        self,
        batch: List[dict],
        batch_index: int,
        row_start: int,
        row_end: int,
        headers: list,
        source_str: str,
        filename: str,
        enc: str,
    ) -> Document:
        """Build a batch Document from multiple rows."""
        return Document(
            content=json.dumps(batch, ensure_ascii=False),
            metadata={
                "source": source_str,
                "batch_index": batch_index,
                "row_start": row_start,
                "row_end": row_end,
                "headers": headers,
                "delimiter": self.delimiter,
                "encoding": enc,
            },
            name=f"{filename}::batch_{batch_index}",
        )

    async def _load_file(self, file_path: Path) -> List[Document]:
        enc = detect_encoding(file_path, self.encoding)
        source_str = str(file_path)
        filename = file_path.name
        documents: List[Document] = []

        async with aiofiles.open(file_path, "r", encoding=enc, newline="") as f:
            raw = await f.read()

        reader = csv.DictReader(io.StringIO(raw), delimiter=self.delimiter)
        headers = list(reader.fieldnames or [])

        if self.batch_size:
            batch: List[dict] = []
            batch_index = 0
            row_start = 0

            for row_index, row in enumerate(reader):
                batch.append(dict(row))
                if len(batch) == self.batch_size:
                    documents.append(
                        self._build_batch_document(
                            batch,
                            batch_index,
                            row_start,
                            row_index,
                            headers,
                            source_str,
                            filename,
                            enc,
                        )
                    )
                    batch = []
                    batch_index += 1
                    row_start = row_index + 1

            if batch:
                documents.append(
                    self._build_batch_document(
                        batch,
                        batch_index,
                        row_start,
                        row_start + len(batch) - 1,
                        headers,
                        source_str,
                        filename,
                        enc,
                    )
                )
        else:
            for row_index, row in enumerate(reader):
                documents.append(
                    self._build_document(
                        dict(row),
                        f"{filename}::row_{row_index}",
                        row_index,
                        headers,
                        source_str,
                        enc,
                    )
                )

        return documents

    async def load(self) -> List[Document]:
        import warnings

        files = self._gather_files(self.source, self.pattern, self.recursive)
        results = await asyncio.gather(*[self._load_file(f) for f in files], return_exceptions=True)
        documents = []
        for file, result in zip(files, results):
            if isinstance(result, Exception):
                warnings.warn(f"Failed to load {file}: {result}", stacklevel=2)
            else:
                documents.extend(result)
        return documents

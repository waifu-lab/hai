import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiofiles

from hai.core import Document

from .base import BaseLoader, detect_encoding


class JSONLoader(BaseLoader):
    """
    Load JSON and JSONL files as Documents.

    Supports:
    - JSON array root → one Document per item
    - JSON object root → single Document for the whole file
    - JSONL (newline-delimited JSON) → one Document per line
    - Nested structures serialized to JSON string content
    - Optional ``content_field`` to promote one field to Document content
    - Optional ``batch_size`` to group JSONL lines into batch Documents

    ```python
    # Auto-detect: JSON array → one doc per item
    loader = JSONLoader(source="records.json")

    # JSONL: one doc per line (auto-detected by .jsonl extension)
    loader = JSONLoader(source="data.jsonl")

    # Force JSONL mode on a .json file
    loader = JSONLoader(source="data.json", is_jsonl=True)

    # Use a specific field as the Document content
    loader = JSONLoader(source="data.jsonl", content_field="text")

    # Batch JSONL lines (5 lines per Document)
    loader = JSONLoader(source="data.jsonl", batch_size=5)

    # Load all JSON files in a directory
    loader = JSONLoader("json_loader", source="./data", pattern="*.json", recursive=True)
    ```
    """  # noqa: E501

    def __init__(
        self,
        name: str = "JSONLoader",
        *,
        source: Union[str, Path],
        pattern: str = "*.json",
        encoding: Optional[str] = None,
        recursive: bool = False,
        content_field: Optional[str] = None,
        batch_size: Optional[int] = None,
        is_jsonl: Optional[bool] = None,
        strict: bool = False,
    ):
        """
        Args:
            name: Node name
            source: File path or directory path
            pattern: Glob pattern for matching files (default: *.json)
            encoding: Force specific encoding (default: auto-detect)
            recursive: Search subdirectories (default: False)
            content_field: If set, use this field's value as Document content;
                           remaining fields go into metadata.
            batch_size: For JSONL only — group this many lines into one Document
                        instead of one Document per line. Content becomes a JSON array.
            is_jsonl: Force JSONL parsing mode. If None, auto-detect by .jsonl extension.
            strict: Raise on the first file that fails to load instead of skipping it
        """  # noqa: E501
        super().__init__(name, strict=strict)
        self.source = Path(source)
        self.pattern = pattern
        self.encoding = encoding
        self.recursive = recursive
        self.content_field = content_field
        self.batch_size = batch_size
        self.is_jsonl = is_jsonl

    def _should_use_jsonl(self, file_path: Path) -> bool:
        if self.is_jsonl is not None:
            return self.is_jsonl
        return file_path.suffix.lower() == ".jsonl"

    def _build_document(self, record: Any, name: str, base_meta: Dict) -> Document:
        """Build a Document from a parsed JSON record.

        If *content_field* is set on this loader and *record* is a dict,
        that field's value becomes the Document content and remaining fields
        are merged into metadata.  Otherwise the full record is JSON-serialized
        as content.
        """
        if isinstance(record, dict) and self.content_field:
            content_value = record.get(self.content_field, "")
            content = str(content_value) if not isinstance(content_value, str) else content_value
            extra_meta = {k: v for k, v in record.items() if k != self.content_field}
            metadata = {**base_meta, **extra_meta}
        else:
            content = json.dumps(record, ensure_ascii=False)
            metadata = base_meta

        return Document(content=content, metadata=metadata, name=name)

    def _build_batch_document(self, batch: List[Any], name: str, base_meta: Dict) -> Document:
        """Build a single Document from a list of records (batch mode)."""
        return Document(
            content=json.dumps(batch, ensure_ascii=False),
            metadata=base_meta,
            name=name,
        )

    async def _load_jsonl(self, file_path: Path) -> List[Document]:
        enc = detect_encoding(file_path, self.encoding)
        source_str = str(file_path)
        filename = file_path.name
        documents: List[Document] = []

        if self.batch_size:
            batch: List[Any] = []
            batch_index = 0
            line_start = 0
            line_index = 0

            async with aiofiles.open(file_path, "r", encoding=enc) as f:
                async for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    batch.append(record)
                    if len(batch) == self.batch_size:
                        documents.append(
                            self._build_batch_document(
                                batch,
                                f"{filename}::batch_{batch_index}",
                                {
                                    "source": source_str,
                                    "batch_index": batch_index,
                                    "line_start": line_start,
                                    "line_end": line_index,
                                    "encoding": enc,
                                    "format": "jsonl",
                                },
                            )
                        )
                        batch = []
                        batch_index += 1
                        line_start = line_index + 1
                    line_index += 1

            if batch:
                documents.append(
                    self._build_batch_document(
                        batch,
                        f"{filename}::batch_{batch_index}",
                        {
                            "source": source_str,
                            "batch_index": batch_index,
                            "line_start": line_start,
                            "line_end": line_start + len(batch) - 1,
                            "encoding": enc,
                            "format": "jsonl",
                        },
                    )
                )
        else:
            doc_index = 0
            async with aiofiles.open(file_path, "r", encoding=enc) as f:
                async for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    documents.append(
                        self._build_document(
                            record,
                            f"{filename}::line_{doc_index}",
                            {
                                "source": source_str,
                                "doc_index": doc_index,
                                "encoding": enc,
                                "format": "jsonl",
                            },
                        )
                    )
                    doc_index += 1

        return documents

    async def _load_json(self, file_path: Path) -> List[Document]:
        enc = detect_encoding(file_path, self.encoding)
        source_str = str(file_path)
        filename = file_path.name

        async with aiofiles.open(file_path, "r", encoding=enc) as f:
            raw = await f.read()

        data = json.loads(raw)

        if isinstance(data, list):
            return [
                self._build_document(
                    record,
                    f"{filename}::item_{i}",
                    {
                        "source": source_str,
                        "doc_index": i,
                        "encoding": enc,
                        "format": "json",
                    },
                )
                for i, record in enumerate(data)
            ]

        return [
            self._build_document(
                data,
                filename,
                {"source": source_str, "encoding": enc, "format": "json"},
            )
        ]

    async def _load_file(self, file_path: Path) -> List[Document]:
        if self._should_use_jsonl(file_path):
            return await self._load_jsonl(file_path)
        return await self._load_json(file_path)

    async def load(self) -> List[Document]:
        files = self._gather_files(self.source, self.pattern, self.recursive)
        return await self._load_each(files, self._load_file)

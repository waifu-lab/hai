import copy
from typing import Any, Dict, Optional


class Document:
    def __init__(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        self.content = content
        self.metadata = copy.deepcopy(metadata) if metadata else {}
        self.name = name or "untitled"

    def get(self, key: str, default: Any = None) -> Any:
        if key == "content":
            return self.content
        elif key == "name":
            return self.name
        elif key in self.metadata:
            return self.metadata[key]
        return default

    def items(self):
        return {**self.metadata, "content": self.content, "name": self.name}.items()

    def __getitem__(self, key: str) -> Any:
        value = self.get(key)
        if value is None and key not in self.metadata:
            raise KeyError(key)
        return value

    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(name='{self.name}', content='{content_preview}')"

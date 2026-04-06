from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseGraphStore(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the graph database."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the connection."""
        pass

    @abstractmethod
    async def execute(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Run a write query (CREATE / MERGE / SET / DELETE).
        No return value — caller is responsible for what it writes.
        """
        pass

    @abstractmethod
    async def query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Run a read query.
        Returns a list of record dicts keyed by the RETURN aliases.
        """
        pass

    @abstractmethod
    async def search(self, index_name: str, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Full-text search via a named database index.
        Returns list of dicts: {"node": {...}, "score": float}
        """
        pass

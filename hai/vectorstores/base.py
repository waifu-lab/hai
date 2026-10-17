from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseVectorStore(ABC):
    @abstractmethod
    async def write(self, data: List[Dict[str, Any]]) -> List[str]:
        """Upsert records and return their ids (generated when a record has no ``id``)."""

    @abstractmethod
    async def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return the *top_k* nearest records as dicts with ``id``, ``score``, ``content``, ``name``, ``payload``."""

    @abstractmethod
    async def get(self, point_id: str) -> Optional[Dict[str, Any]]:
        """Return ``{"id", "payload"}`` for one record, or None if it does not exist."""

    @abstractmethod
    async def delete(self, point_id: str) -> None:
        """Delete one record."""

    @abstractmethod
    async def delete_collection(self) -> None:
        """Delete every record together with the collection itself."""

    @abstractmethod
    async def close(self) -> None:
        """Release connections held by the store."""

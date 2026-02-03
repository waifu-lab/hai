from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseGraphStore(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to graph database."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connection to graph database."""
        pass

    @abstractmethod
    async def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a new fact (triplet) to the graph.

        Args:
            subject: Subject entity
            predicate: Relationship type
            obj: Object entity
            confidence: Confidence score (0-1)
            source: Source document/chunk ID
            metadata: Additional metadata

        Returns:
            fact_id: Unique identifier for the fact
        """
        pass

    @abstractmethod
    async def get_fact(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a fact by ID.

        Returns:
            Fact data including status, timestamps, etc.
        """
        pass

    @abstractmethod
    async def query_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Query facts by triplet pattern.

        Args:
            subject: Filter by subject (None = any)
            predicate: Filter by predicate (None = any)
            obj: Filter by object (None = any)
            limit: Maximum results

        Returns:
            List of matching facts
        """
        pass

    @abstractmethod
    async def get_entity_relations(
        self, entity: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get all relationships for an entity.

        Returns:
            List of facts involving this entity
        """
        pass

    @abstractmethod
    async def supersede_fact(self, old_fact_id: str, new_fact_id: str) -> None:
        """
        Mark an old fact as superseded by a new one.
        Creates a merge_by relationship.

        Args:
            old_fact_id: ID of fact being replaced
            new_fact_id: ID of replacement fact
        """
        pass

    @abstractmethod
    async def get_fact_history(self, fact_id: str) -> List[Dict[str, Any]]:
        """
        Get version history of a fact.

        Returns:
            List of facts in the version chain
        """
        pass

    @abstractmethod
    async def mark_conflicted(self, fact_ids: List[str], reason: str) -> None:
        """
        Mark multiple facts as conflicted.

        Args:
            fact_ids: IDs of conflicting facts
            reason: Reason for conflict
        """
        pass

    @abstractmethod
    async def add_entity(
        self,
        name: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add or update an entity.

        Args:
            name: Entity name
            entity_type: Type (Person, Organization, etc.)
            properties: Entity properties

        Returns:
            entity_id: Unique identifier
        """
        pass

    @abstractmethod
    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get entity by ID.

        Returns:
            Entity data with properties
        """
        pass

    @abstractmethod
    async def search_entities(
        self, name: str, entity_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search entities by name and type.

        Args:
            name: Entity name to search
            entity_type: Optional type filter

        Returns:
            List of matching entities
        """
        pass

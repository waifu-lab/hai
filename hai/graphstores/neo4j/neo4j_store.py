import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from hai.graphstores.base import BaseGraphStore


class Neo4jStore(BaseGraphStore):
    def __init__(
        self,
        name: str = "Neo4jStore",
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ):
        super().__init__(name)
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None

    async def connect(self) -> None:
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as exc:
            msg = "neo4j package required. Install: pip install neo4j"
            raise ImportError(msg) from exc

        self.driver = AsyncGraphDatabase.driver(
            self.uri, auth=(self.username, self.password)
        )

        async with self.driver.session(database=self.database) as session:
            await session.run("RETURN 1")

    async def close(self) -> None:
        if self.driver:
            await self.driver.close()

    async def add_entity(
        self,
        name: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        entity_id = str(uuid.uuid4())
        props = properties or {}
        props.update(
            {
                "id": entity_id,
                "name": name,
                "type": entity_type,
                "created_at": datetime.utcnow().isoformat(),
            }
        )

        query = """
        MERGE (e:Entity {name: $name, type: $type})
        ON CREATE SET e = $props
        ON MATCH SET e.updated_at = $timestamp
        RETURN e.id as id
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                query,
                name=name,
                type=entity_type,
                props=props,
                timestamp=datetime.utcnow().isoformat(),
            )
            record = await result.single()
            return record["id"] if record else entity_id

    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by ID."""
        query = """
        MATCH (e:Entity {id: $entity_id})
        RETURN e
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, entity_id=entity_id)
            record = await result.single()
            if record:
                return dict(record["e"])
            return None

    async def search_entities(
        self, name: str, entity_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if entity_type:
            query = """
            MATCH (e:Entity)
            WHERE e.name CONTAINS $name AND e.type = $entity_type
            RETURN e
            LIMIT 20
            """
            params = {"name": name, "entity_type": entity_type}
        else:
            query = """
            MATCH (e:Entity)
            WHERE e.name CONTAINS $name
            RETURN e
            LIMIT 20
            """
            params = {"name": name}

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, **params)
            records = await result.data()
            return [dict(record["e"]) for record in records]

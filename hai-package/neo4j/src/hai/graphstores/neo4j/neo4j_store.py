from typing import Any, Dict, List, Optional

from hai.graphstores import BaseGraphStore
from neo4j import AsyncDriver, AsyncGraphDatabase


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
        self.driver: Optional[AsyncDriver] = None

    def _get_driver(self) -> AsyncDriver:
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized. Call connect() first.")
        return self.driver

    async def connect(self) -> None:
        self.driver = AsyncGraphDatabase.driver(self.uri, auth=(self.username, self.password))
        async with self._get_driver().session(database=self.database) as session:
            await session.run("RETURN 1")

    async def close(self) -> None:
        if self.driver:
            await self.driver.close()

    async def execute(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        driver = self._get_driver()
        async with driver.session(database=self.database) as session:
            await session.run(query=query, parameters=parameters or {})  # type: ignore[arg-type]

    async def query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        driver = self._get_driver()
        async with driver.session(database=self.database) as session:
            result = await session.run(query=query, parameters=parameters or {})  # type: ignore[arg-type]
            return await result.data()

    async def search(self, index_name: str, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        cypher = """
        CALL db.index.fulltext.queryNodes($index_name, $query_text)
        YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        LIMIT $limit
        """
        driver = self._get_driver()
        async with driver.session(database=self.database) as session:
            result = await session.run(
                query=cypher,
                parameters={
                    "index_name": index_name,
                    "query_text": query_text,
                    "limit": limit,
                },
            )
            records = await result.data()
            return [{"node": dict(r["node"]), "score": r["score"]} for r in records]

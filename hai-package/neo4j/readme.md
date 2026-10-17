# hydrogen-ai-neo4j

Neo4j graph store for [hydrogen-ai](../../README.md).

```bash
pip install hydrogen-ai-neo4j
```

```python
from hai.graphstores.neo4j import Neo4jStore

store = Neo4jStore(uri="bolt://localhost:7687", username="neo4j", password="...")
await store.connect()
records = await store.query("MATCH (n) RETURN n LIMIT 5")
await store.close()
```

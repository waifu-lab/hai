# hydrogen-ai-qdrant

Qdrant vector store for [hydrogen-ai](../../README.md).

```bash
pip install hydrogen-ai-qdrant
```

```python
from hai.vectorstores import VectorStoreRetriever, VectorStoreWriter
from hai.vectorstores.qdrant import QdrantStore

store = QdrantStore(url="http://localhost:6333", collection_name="docs")
# or a local file / in-memory instance:
# store = QdrantStore(path="./qdrant_data")
# store = QdrantStore(location=":memory:")

writer = VectorStoreWriter(store=store)
retriever = VectorStoreRetriever(store=store, top_k=5)
```

The collection is created on first use. Its vector size is inferred from the first
embedding unless `embedding_dim` is given.

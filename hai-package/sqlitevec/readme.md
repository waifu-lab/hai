# hydrogen-ai-sqlitevec

Local vector store for [hydrogen-ai](../../README.md) in a single SQLite file, using
[sqlite-vec](https://github.com/asg017/sqlite-vec). No server needed.

```bash
pip install hydrogen-ai-sqlitevec
```

```python
from hai.vectorstores import VectorStoreRetriever, VectorStoreWriter
from hai.vectorstores.sqlitevec import SqliteVecStore

store = SqliteVecStore(path="hai.db", collection_name="docs")

writer = VectorStoreWriter(store=store)
retriever = VectorStoreRetriever(store=store, top_k=5)
```

The tables are created on first use. Their vector size is inferred from the first
embedding unless `embedding_dim` is given.

sqlite-vec is a SQLite extension, so Python's `sqlite3` must be able to load
extensions. Builds from python.org, Homebrew, uv and pyenv can; the Python that
ships with macOS cannot.

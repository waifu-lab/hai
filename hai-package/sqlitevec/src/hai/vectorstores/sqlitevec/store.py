import asyncio
import json
import re
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, TypeVar

import sqlite_vec

from hai.vectorstores import BaseVectorStore

T = TypeVar("T")

_METRICS = {"cosine": "cosine", "euclid": "l2", "euclidean": "l2", "l2": "l2", "l1": "l1"}
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VECTOR_DIM = re.compile(r"float\[(\d+)\]")


class SqliteVecStore(BaseVectorStore):
    """
    Records live in two tables: ``{collection_name}`` (id, content, name, JSON payload)
    and the ``vec0`` virtual table ``{collection_name}_vec`` holding the vectors. Both
    are created on the first :meth:`write` or :meth:`query`; the vector size is
    *embedding_dim*, or the length of the first vector seen when None. Vectors whose
    length differs from the collection's raise ``ValueError``.

    ``score`` in query results follows Qdrant: for ``cosine`` it is the cosine
    similarity (higher is closer) and ``score_threshold`` is a minimum; for ``euclid``
    and ``l1`` it is the distance (lower is closer) and ``score_threshold`` is a maximum.

    SQLite is synchronous, so all database work runs on one dedicated worker thread
    that owns the connection; awaiting the store never blocks the event loop.
    """

    def __init__(
        self,
        path: str = "hai.db",
        collection_name: str = "Document",
        embedding_dim: Optional[int] = None,
        similarity: Literal["cosine", "euclid", "euclidean", "l2", "l1"] = "cosine",
    ):
        """
        Args:
            path: SQLite database file, or ``":memory:"``.
            collection_name: Table name prefix; must be a plain SQL identifier.
            embedding_dim: Vector size of a new collection. None infers it from the first vector.
            similarity: Distance metric: ``"cosine"``, ``"euclid"`` (aliases ``"euclidean"``, ``"l2"``) or ``"l1"``.
        """
        if not _IDENTIFIER.match(collection_name):
            raise ValueError(f"collection_name must be a plain SQL identifier, got '{collection_name}'.")
        if similarity.lower() not in _METRICS:
            raise ValueError(f"Unknown similarity '{similarity}'. Choose from {sorted(_METRICS)}.")
        self.path = path
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.similarity = similarity
        self._metric = _METRICS[similarity.lower()]
        self._docs_table = collection_name
        self._vec_table = f"{collection_name}_vec"
        self._executor: Optional[ThreadPoolExecutor] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._collection_dim: Optional[int] = None

    async def _run(self, fn: Callable[..., T], *args: Any) -> T:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sqlitevec")
        return await asyncio.get_running_loop().run_in_executor(self._executor, partial(fn, *args))

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(self.path)
            try:
                conn.enable_load_extension(True)
            except AttributeError as e:
                conn.close()
                raise RuntimeError(
                    "This Python's sqlite3 module cannot load extensions, which sqlite-vec requires. "
                    "Use a Python build with extension loading enabled (python.org, Homebrew, uv, pyenv)."
                ) from e
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self._conn = conn
        return self._conn

    def _has_tables(self, conn: sqlite3.Connection) -> bool:
        row = conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (self._vec_table,)).fetchone()
        return row is not None

    def _ensure_tables(self, vector_dim: int) -> sqlite3.Connection:
        """Create the tables if missing and check that *vector_dim* matches the collection."""
        conn = self._connect()
        if self._collection_dim is None:
            row = conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (self._vec_table,)).fetchone()
            if row is not None:
                self._collection_dim = int(_VECTOR_DIM.search(row[0]).group(1))
            else:
                dim = self.embedding_dim or vector_dim
                with conn:
                    conn.execute(
                        f"CREATE TABLE IF NOT EXISTS {self._docs_table} ("
                        "pk INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, content TEXT, name TEXT, payload TEXT)"
                    )
                    conn.execute(
                        f"CREATE VIRTUAL TABLE {self._vec_table} "
                        f"USING vec0(embedding float[{dim}] distance_metric={self._metric})"
                    )
                self._collection_dim = dim

        if vector_dim != self._collection_dim:
            raise ValueError(
                f"Vector size {vector_dim} does not match collection '{self.collection_name}' "
                f"(size {self._collection_dim}). Use the same embedding model, or another collection_name."
            )
        return conn

    def _write(self, data: List[Dict[str, Any]]) -> List[str]:
        dims = {len(item["embedding"]) for item in data}
        if len(dims) > 1:
            raise ValueError(f"All embeddings in one write must have the same size, got {sorted(dims)}.")
        conn = self._ensure_tables(dims.pop())

        ids = []
        with conn:
            for item in data:
                point_id = str(item.get("id") or uuid.uuid4())
                payload = {
                    "id": point_id,
                    "content": item.get("content", ""),
                    "name": item.get("name", ""),
                    **{k: v for k, v in item.items() if k not in ("id", "content", "name", "embedding")},
                }
                conn.execute(
                    f"INSERT INTO {self._docs_table} (id, content, name, payload) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "content = excluded.content, name = excluded.name, payload = excluded.payload",
                    (point_id, payload["content"], payload["name"], json.dumps(payload, ensure_ascii=False)),
                )
                (pk,) = conn.execute(f"SELECT pk FROM {self._docs_table} WHERE id = ?", (point_id,)).fetchone()
                conn.execute(f"DELETE FROM {self._vec_table} WHERE rowid = ?", (pk,))
                conn.execute(
                    f"INSERT INTO {self._vec_table} (rowid, embedding) VALUES (?, ?)",
                    (pk, sqlite_vec.serialize_float32(item["embedding"])),
                )
                ids.append(point_id)
        return ids

    def _query(self, embedding: List[float], top_k: int, score_threshold: Optional[float]) -> List[Dict[str, Any]]:
        conn = self._ensure_tables(len(embedding))
        rows = conn.execute(
            f"SELECT d.id, d.content, d.name, d.payload, v.distance "
            f"FROM (SELECT rowid, distance FROM {self._vec_table} WHERE embedding MATCH ? AND k = ?) AS v "
            f"JOIN {self._docs_table} AS d ON d.pk = v.rowid ORDER BY v.distance",
            (sqlite_vec.serialize_float32(embedding), top_k),
        ).fetchall()

        results = []
        for point_id, content, name, payload, distance in rows:
            if self._metric == "cosine":
                score = 1.0 - distance
                if score_threshold is not None and score < score_threshold:
                    continue
            else:
                score = distance
                if score_threshold is not None and score > score_threshold:
                    continue
            results.append(
                {"id": point_id, "score": score, "content": content, "name": name, "payload": json.loads(payload)}
            )
        return results

    def _get(self, point_id: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        if not self._has_tables(conn):
            return None
        row = conn.execute(f"SELECT id, payload FROM {self._docs_table} WHERE id = ?", (point_id,)).fetchone()
        return {"id": row[0], "payload": json.loads(row[1])} if row else None

    def _delete(self, point_id: str) -> None:
        conn = self._connect()
        if not self._has_tables(conn):
            return
        with conn:
            row = conn.execute(f"SELECT pk FROM {self._docs_table} WHERE id = ?", (point_id,)).fetchone()
            if row:
                conn.execute(f"DELETE FROM {self._vec_table} WHERE rowid = ?", row)
                conn.execute(f"DELETE FROM {self._docs_table} WHERE pk = ?", row)

    def _delete_collection(self) -> None:
        conn = self._connect()
        with conn:
            conn.execute(f"DROP TABLE IF EXISTS {self._vec_table}")
            conn.execute(f"DROP TABLE IF EXISTS {self._docs_table}")
        self._collection_dim = None

    def _collection_info(self) -> Dict[str, Any]:
        conn = self._connect()
        count = conn.execute(f"SELECT COUNT(*) FROM {self._docs_table}").fetchone()[0] if self._has_tables(conn) else 0
        return {"name": self.collection_name, "points_count": count, "dim": self._collection_dim}

    def _close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def write(self, data: List[Dict[str, Any]]) -> List[str]:
        if not data:
            return []
        return await self._run(self._write, data)

    async def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        return await self._run(self._query, query_embedding, top_k, score_threshold)

    async def get(self, point_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single record by ID. Returns None if not found."""
        return await self._run(self._get, point_id)

    async def delete(self, point_id: str) -> None:
        """Delete a single record by ID."""
        await self._run(self._delete, point_id)

    async def delete_collection(self) -> None:
        await self._run(self._delete_collection)

    async def get_collection_info(self) -> Dict[str, Any]:
        return await self._run(self._collection_info)

    async def close(self) -> None:
        if self._executor is not None:
            await self._run(self._close)
            self._executor.shutdown(wait=True)
            self._executor = None

import logging
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

from hai.vectorstores import BaseVectorStore

logger = logging.getLogger(__name__)

_DISTANCES = {
    "cosine": Distance.COSINE,
    "euclid": Distance.EUCLID,
    "euclidean": Distance.EUCLID,
    "dot": Distance.DOT,
}
_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "hydrogen-ai-qdrant")


def _to_point_id(record_id: Union[str, int]) -> Union[str, int]:
    """Qdrant only accepts unsigned ints and UUIDs as point ids; any other id maps to a stable UUID5."""
    if isinstance(record_id, int):
        return record_id
    try:
        return str(uuid.UUID(str(record_id)))
    except ValueError:
        return str(uuid.uuid5(_ID_NAMESPACE, str(record_id)))


class QdrantStore(BaseVectorStore):
    """Qdrant collection as a :class:`~hai.vectorstores.BaseVectorStore`.

    The collection is created on the first :meth:`write` or :meth:`query`. Its vector
    size is *embedding_dim* when given, otherwise the length of the first vector seen,
    so it always matches the embedding model in use. Vectors whose length differs from
    the collection's raise ``ValueError`` instead of failing inside Qdrant.

    Record ids can be any string. Ids that are not UUIDs or integers are stored as a
    UUID derived from the id, with the original kept in the payload under ``id``;
    results, :meth:`get` and :meth:`delete` all use the original id.
    """

    def __init__(
        self,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[int] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
        force_disable_check_same_thread: bool = False,
        collection_name: str = "Document",
        embedding_dim: Optional[int] = None,
        content_field: str = "content",
        name_field: str = "name",
        embedding_field: str = "embedding",
        similarity: Literal["cosine", "euclid", "euclidean", "dot"] = "cosine",
        shard_number: Optional[int] = None,
        replication_factor: Optional[int] = None,
        write_consistency_factor: Optional[int] = None,
        on_disk_payload: Optional[bool] = None,
        hnsw_config: Optional[dict] = None,
        optimizers_config: Optional[dict] = None,
        wal_config: Optional[dict] = None,
        quantization_config: Optional[dict] = None,
        init_from: Optional[dict] = None,
    ):
        """
        Args:
            embedding_dim: Vector size of a new collection. None infers it from the first vector.
            similarity: Distance metric: ``"cosine"``, ``"euclid"`` (alias ``"euclidean"``) or ``"dot"``.
        """
        if similarity.lower() not in _DISTANCES:
            raise ValueError(f"Unknown similarity '{similarity}'. Choose from {sorted(_DISTANCES)}.")
        self._client: Optional[AsyncQdrantClient] = None
        self._collection_dim: Optional[int] = None
        self.location = location
        self.url = url
        self.port = port
        self.grpc_port = grpc_port
        self.prefer_grpc = prefer_grpc
        self.https = https
        self.api_key = api_key
        self.prefix = prefix
        self.timeout = timeout
        self.host = host
        self.path = path
        self.force_disable_check_same_thread = force_disable_check_same_thread
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.content_field = content_field
        self.name_field = name_field
        self.embedding_field = embedding_field
        self.similarity = similarity
        self.shard_number = shard_number
        self.replication_factor = replication_factor
        self.write_consistency_factor = write_consistency_factor
        self.on_disk_payload = on_disk_payload
        self.hnsw_config = hnsw_config
        self.optimizers_config = optimizers_config
        self.wal_config = wal_config
        self.quantization_config = quantization_config
        self.init_from = init_from

    def _get_client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(
                location=self.location,
                url=self.url,
                port=self.port,
                grpc_port=self.grpc_port,
                prefer_grpc=self.prefer_grpc,
                https=self.https,
                prefix=self.prefix,
                api_key=self.api_key,
                timeout=self.timeout,
                host=self.host,
                path=self.path,
                force_disable_check_same_thread=self.force_disable_check_same_thread,
            )
        return self._client

    async def _ensure_collection(self, vector_dim: int) -> AsyncQdrantClient:
        """Create the collection if missing and check that *vector_dim* matches it."""
        client = self._get_client()
        if self._collection_dim is None:
            if await client.collection_exists(self.collection_name):
                info = await client.get_collection(self.collection_name)
                vectors = info.config.params.vectors
                self._collection_dim = vectors.size if isinstance(vectors, VectorParams) else None
                logger.debug("collection '%s' already exists", self.collection_name)
            else:
                self._collection_dim = self.embedding_dim or vector_dim
                await self._create_collection(self._collection_dim)

        if self._collection_dim is not None and vector_dim != self._collection_dim:
            raise ValueError(
                f"Vector size {vector_dim} does not match collection '{self.collection_name}' "
                f"(size {self._collection_dim}). Use the same embedding model, or another collection_name."
            )
        return client

    async def _create_collection(self, dim: int) -> None:
        create_params: Dict[str, Any] = {
            "collection_name": self.collection_name,
            "vectors_config": VectorParams(size=dim, distance=_DISTANCES[self.similarity.lower()]),
        }
        optional_params = {
            "shard_number": self.shard_number,
            "replication_factor": self.replication_factor,
            "write_consistency_factor": self.write_consistency_factor,
            "on_disk_payload": self.on_disk_payload,
            "hnsw_config": self.hnsw_config,
            "optimizers_config": self.optimizers_config,
            "wal_config": self.wal_config,
            "quantization_config": self.quantization_config,
            "init_from": self.init_from,
        }
        create_params.update({k: v for k, v in optional_params.items() if v is not None})
        await self._get_client().create_collection(**create_params)

    async def write(self, data: List[Dict[str, Any]]) -> List[str]:
        if not data:
            return []
        dims = {len(item["embedding"]) for item in data}
        if len(dims) > 1:
            raise ValueError(f"All embeddings in one write must have the same size, got {sorted(dims)}.")
        client = await self._ensure_collection(dims.pop())

        points = []
        ids = []

        for item in data:
            record_id = item.get("id") or str(uuid.uuid4())
            ids.append(record_id)

            payload = {
                "id": record_id,
                self.content_field: item.get("content", ""),
                self.name_field: item.get("name", ""),
            }

            for key, value in item.items():
                if key not in ["id", "content", "name", "embedding"]:
                    payload[key] = value

            point = PointStruct(
                id=_to_point_id(record_id),
                vector=item["embedding"],
                payload=payload,
            )
            points.append(point)

        await client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        return ids

    async def _query_by_embedding(
        self,
        embedding: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        client = await self._ensure_collection(len(embedding))

        search_result = await client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=filter_conditions,
        )

        results = []
        for scored_point in search_result.points:
            result = {
                "id": scored_point.payload.get("id", scored_point.id),
                "score": scored_point.score,
                "content": scored_point.payload.get(self.content_field, ""),
                "name": scored_point.payload.get(self.name_field, ""),
                "payload": scored_point.payload,
            }
            results.append(result)

        return results

    async def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        return await self._query_by_embedding(
            embedding=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
        )

    async def get_collection_info(self) -> Dict[str, Any]:
        info = await self._get_client().get_collection(collection_name=self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "status": info.status,
        }

    async def get(self, point_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single point by ID. Returns None if not found."""
        client = self._get_client()
        if not await client.collection_exists(self.collection_name):
            return None
        results = await client.retrieve(
            collection_name=self.collection_name,
            ids=[_to_point_id(point_id)],
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return None
        point = results[0]
        return {"id": point.payload.get("id", point.id), "payload": point.payload}

    async def delete(self, point_id: str) -> None:
        """Delete a single point by ID."""
        client = self._get_client()
        if not await client.collection_exists(self.collection_name):
            return
        await client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[_to_point_id(point_id)]),
        )

    async def delete_collection(self) -> None:
        await self._get_client().delete_collection(collection_name=self.collection_name)
        self._collection_dim = None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

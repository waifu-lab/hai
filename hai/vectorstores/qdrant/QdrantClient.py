import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantStore:
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
        embedding_dim: int = 768,
        content_field: str = "content",
        name_field: str = "name",
        embedding_field: str = "embedding",
        similarity: str = "cosine",
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
        self._client = None
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

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
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

            self._init_collection()
        return self._client

    def _init_collection(self):
        client = self._client

        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]

        if self.collection_name not in collection_names:
            distance_map = {
                "cosine": Distance.COSINE,
                "euclid": Distance.EUCLID,
                "dot": Distance.DOT,
                "euclidean": Distance.EUCLID,
            }
            distance = distance_map.get(self.similarity.lower(), Distance.COSINE)

            create_params = {
                "collection_name": self.collection_name,
                "vectors_config": VectorParams(
                    size=self.embedding_dim,
                    distance=distance,
                ),
            }

            if self.shard_number is not None:
                create_params["shard_number"] = self.shard_number
            if self.replication_factor is not None:
                create_params["replication_factor"] = self.replication_factor
            if self.write_consistency_factor is not None:
                create_params["write_consistency_factor"] = (
                    self.write_consistency_factor
                )
            if self.on_disk_payload is not None:
                create_params["on_disk_payload"] = self.on_disk_payload
            if self.hnsw_config is not None:
                create_params["hnsw_config"] = self.hnsw_config
            if self.optimizers_config is not None:
                create_params["optimizers_config"] = self.optimizers_config
            if self.wal_config is not None:
                create_params["wal_config"] = self.wal_config
            if self.quantization_config is not None:
                create_params["quantization_config"] = self.quantization_config
            if self.init_from is not None:
                create_params["init_from"] = self.init_from

            client.create_collection(**create_params)

        else:
            print(f"Collection '{self.collection_name}' already exists.")

    def write(self, data: List[Dict[str, Any]]) -> List[str]:
        client = self._get_client()

        points = []
        ids = []

        for item in data:
            point_id = item.get("id", str(uuid.uuid4()))
            ids.append(point_id)

            payload = {
                self.content_field: item.get("content", ""),
                self.name_field: item.get("name", ""),
            }

            for key, value in item.items():
                if key not in ["id", "content", "name", "embedding"]:
                    payload[key] = value

            point = PointStruct(
                id=point_id,
                vector=item["embedding"],
                payload=payload,
            )
            points.append(point)

        client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        return ids

    def _query_by_embedding(
        self,
        embedding: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        client = self._get_client()

        search_result = client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=filter_conditions,
        )

        results = []
        for scored_point in search_result.points:
            result = {
                "id": scored_point.id,
                "score": scored_point.score,
                "content": scored_point.payload.get(self.content_field, ""),
                "name": scored_point.payload.get(self.name_field, ""),
                "payload": scored_point.payload,
            }
            results.append(result)

        return results

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        return self._query_by_embedding(
            embedding=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
        )

    def get_collection_info(self) -> Dict[str, Any]:
        client = self._get_client()
        info = client.get_collection(collection_name=self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "status": info.status,
        }

    def delete_collection(self):
        client = self._get_client()
        client.delete_collection(collection_name=self.collection_name)

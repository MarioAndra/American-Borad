from __future__ import annotations

from typing import Any

import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import MetadataQuery

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

WEAVIATE_CLASS = "KnowledgeChunk"


class WeaviateService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: weaviate.WeaviateClient | None = None
        self._connect()

    def _connect(self) -> None:
        url = self.settings.WEAVIATE_URL
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            host = parsed.hostname or "localhost"
            http_port = parsed.port or 8080
            grpc_port = http_port + 1 if parsed.port else 50051
            self._client = weaviate.connect_to_custom(
                http_host=host,
                http_port=http_port,
                http_secure=parsed.scheme == "https",
                grpc_host=host,
                grpc_port=grpc_port,
                grpc_secure=False,
                skip_init_checks=True,
            )
            if not self._client.is_connected():
                log.warning("Weaviate server not reachable at %s", url)
                self._client = None
                return
            self._ensure_schema()
            log.info("Connected to Weaviate at %s", url)
        except Exception as exc:
            log.warning("Failed to connect to Weaviate: %s", exc)
            self._client = None

    @property
    def is_available(self) -> bool:
        return self._client is not None and self._client.is_connected()

    def _ensure_schema(self) -> None:
        if self._client.collections.exists(WEAVIATE_CLASS):
            return
        self._client.collections.create(
            name=WEAVIATE_CLASS,
            description="Chunks of text from course PDFs with embeddings for RAG",
            properties=[
                wvc.config.Property(name="chunk_id", data_type=wvc.config.DataType.INT),
                wvc.config.Property(name="document_id", data_type=wvc.config.DataType.INT),
                wvc.config.Property(name="course_name", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="chunk_index", data_type=wvc.config.DataType.INT),
                wvc.config.Property(name="topic_id", data_type=wvc.config.DataType.INT),
            ],
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        )
        log.info("Created Weaviate class %s", WEAVIATE_CLASS)

    # ── Ingestion ───────────────────────────────────────────────

    def store_chunk(
        self,
        chunk_id: int,
        document_id: int,
        course_name: str,
        title: str,
        text: str,
        chunk_index: int,
        topic_id: int | None,
        embedding: list[float],
    ) -> bool:
        """Store a chunk vector in Weaviate. Returns True on success, False on failure."""
        if not self.is_available:
            log.warning("Weaviate unavailable — skipping store for chunk %d", chunk_id)
            return False
        try:
            collection = self._client.collections.get(WEAVIATE_CLASS)
            collection.data.insert(
                properties={
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "course_name": course_name,
                    "title": title,
                    "chunk_index": chunk_index,
                    "topic_id": topic_id,
                },
                vector=embedding,
            )
            return True
        except Exception as exc:
            log.error("Failed to store chunk %d in Weaviate: %s", chunk_id, exc)
            return False

    # ── Retrieval ───────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        topic_id: int,
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        if not self.is_available:
            log.warning("Weaviate unavailable — returning empty search results")
            return []
        try:
            collection = self._client.collections.get(WEAVIATE_CLASS)
            response = collection.query.near_vector(
                near_vector=query_embedding,
                filters=wvc.query.Filter.by_property("topic_id").equal(topic_id),
                limit=top_k,
                return_metadata=MetadataQuery(distance=True),
            )
            results = []
            for obj in response.objects:
                props = obj.properties
                results.append({
                    "chunk_id": props["chunk_id"],
                    "document_id": props["document_id"],
                    "course_name": props.get("course_name", ""),
                    "title": props.get("title", ""),
                    "chunk_index": props.get("chunk_index", 0),
                    "topic_id": props.get("topic_id"),
                    "similarity": 1.0 - obj.metadata.distance if obj.metadata and obj.metadata.distance is not None else None,
                })
            return results
        except Exception as exc:
            log.error("Weaviate search failed: %s", exc)
            return []

    def count_objects(self) -> tuple[int, int]:
        """Count total Weaviate objects.
        Returns (total_objects, 0) — orphan count requires full scan which is heavy."""
        if not self.is_available:
            return 0, 0
        try:
            collection = self._client.collections.get(WEAVIATE_CLASS)
            total = collection.aggregate.over_all(total_count=True)
            total_count = total.total_count if total and total.total_count else 0
            return total_count, 0
        except Exception as exc:
            log.error("Failed to count Weaviate objects: %s", exc)
            return 0, 0

    def chunk_exists(self, chunk_id: int) -> bool:
        """Check if a specific chunk exists in Weaviate by chunk_id using a filtered aggregate."""
        if not self.is_available:
            return False
        try:
            collection = self._client.collections.get(WEAVIATE_CLASS)
            agg = collection.aggregate.over_all(
                filters=wvc.query.Filter.by_property("chunk_id").equal(chunk_id),
                total_count=True,
            )
            count = agg.total_count if agg and agg.total_count else 0
            return count > 0
        except Exception:
            return False

    def delete_document_chunks(self, document_id: int) -> None:
        """Delete all Weaviate objects for a given document_id (enables idempotent re-embed)."""
        if not self.is_available:
            return
        try:
            collection = self._client.collections.get(WEAVIATE_CLASS)
            collection.data.delete_many(
                where=wvc.query.Filter.by_property("document_id").equal(document_id),
            )
        except Exception as exc:
            log.warning("Failed to clear Weaviate objects for document %d: %s", document_id, exc)

    def close(self) -> None:
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass

from __future__ import annotations

from dataclasses import dataclass
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models import KnowledgeChunk, Topic
from app.services.llm_client import LLMClient
from app.services.weaviate_client import WeaviateService

log = get_logger(__name__)



@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    course_name: str
    title: str
    text: str
    chunk_index: int
    topic_id: int | None
    similarity: float | None


class RAGRetrievalService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()
        self._llm = LLMClient(provider=self.settings.RAG_EMBEDDING_PROVIDER)
        self._weaviate = WeaviateService()

    def retrieve(
        self,
        topic_id: int,
        query: str | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.RAG_RETRIEVAL_TOP_K
        min_similarity = min_similarity if min_similarity is not None else self.settings.RAG_MIN_SIMILARITY

        if query and self._llm.is_available:
            return self._retrieve_vector(topic_id, query, top_k, min_similarity)

        if not query or not self._llm.is_available:
            if self.settings.RAG_ALLOW_VECTOR_FALLBACK:
                return self._retrieve_random(topic_id, top_k)
            return []

    def retrieve_multi(
        self,
        topic_id: int,
        queries: list[str],
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve using multiple queries, merging and deduplicating results.

        Embeds all queries in a single batch, searches Weaviate per
        embedding, then merges by ``chunk_id`` keeping the best
        similarity score.  Falls back to single-query retrieval on
        partial failure.
        """
        top_k = top_k or self.settings.RAG_RETRIEVAL_TOP_K
        min_similarity = (
            min_similarity
            if min_similarity is not None
            else self.settings.RAG_MIN_SIMILARITY
        )

        valid_queries = [q for q in queries if q and q.strip()]
        if not valid_queries:
            return []

        if not self._llm.is_available:
            if self.settings.RAG_ALLOW_VECTOR_FALLBACK:
                return self._retrieve_random(topic_id, top_k)
            return []

        # Batch-embed all queries
        embeddings = self._llm.embed(valid_queries)
        query_pairs: list[tuple[str, list[float]]] = []
        for q, emb in zip(valid_queries, embeddings):
            if emb is not None:
                query_pairs.append((q, emb))

        if not query_pairs:
            log.warning("Multi-query: all embeddings failed — falling back to single-query")
            return self._retrieve_vector(topic_id, valid_queries[0], top_k, min_similarity)

        # Search Weaviate per embedding and merge
        merged: dict[int, tuple[RetrievedChunk, float]] = {}
        for q_text, emb in query_pairs:
            try:
                weaviate_results = self._weaviate.search(
                    query_embedding=emb,
                    topic_id=topic_id,
                    top_k=top_k,
                )
            except Exception as exc:
                log.warning("Multi-query: Weaviate search failed for %r: %s", q_text[:60], exc)
                continue
            if not weaviate_results:
                continue

            chunk_ids = [r["chunk_id"] for r in weaviate_results]
            sim_map = {r["chunk_id"]: r["similarity"] for r in weaviate_results}

            rows = (
                self.db.query(KnowledgeChunk)
                .filter(KnowledgeChunk.id.in_(chunk_ids))
                .all()
            )
            row_map = {c.id: c for c in rows}

            for cid in chunk_ids:
                c = row_map.get(cid)
                if c is None:
                    continue
                sim = sim_map.get(cid)
                if sim is not None and sim < min_similarity:
                    continue
                chunk = RetrievedChunk(
                    chunk_id=c.id,
                    document_id=c.document_id,
                    course_name=c.document.course_name if c.document else "",
                    title=c.document.title if c.document else "",
                    text=c.text,
                    chunk_index=c.chunk_index,
                    topic_id=c.topic_id,
                    similarity=sim,
                )
                existing = merged.get(cid)
                if existing is None or (sim or 0) > existing[1]:
                    merged[cid] = (chunk, sim or 0)

        if not merged:
            if self.settings.RAG_ALLOW_VECTOR_FALLBACK:
                log.warning("Multi-query: no chunks found — falling back to random")
                return self._retrieve_random(topic_id, top_k)
            return []

        # Sort by similarity descending, cap at top_k
        results = [chunk for chunk, _ in sorted(merged.values(), key=lambda x: x[1], reverse=True)]
        return results[:top_k]

    def _retrieve_vector(
        self,
        topic_id: int,
        query: str,
        top_k: int,
        min_similarity: float,
    ) -> list[RetrievedChunk]:
        embeddings = self._llm.embed([query])
        query_embedding = embeddings[0]
        if query_embedding is None:
            log.error("Embedding failed for retrieval query")
            if self.settings.RAG_ALLOW_VECTOR_FALLBACK:
                return self._retrieve_random(topic_id, top_k)
            return []

        weaviate_results = self._weaviate.search(
            query_embedding=query_embedding,
            topic_id=topic_id,
            top_k=top_k,
        )
        if not weaviate_results:
            log.warning("Weaviate returned no results")
            if self.settings.RAG_ALLOW_VECTOR_FALLBACK:
                log.warning("RAG_ALLOW_VECTOR_FALLBACK enabled — falling back to random PG chunks")
                return self._retrieve_random(topic_id, top_k)
            log.warning("RAG_ALLOW_VECTOR_FALLBACK disabled — returning empty (generation will skip)")
            return []

        chunk_ids = [r["chunk_id"] for r in weaviate_results]
        sim_map = {r["chunk_id"]: r["similarity"] for r in weaviate_results}

        rows = (
            self.db.query(KnowledgeChunk)
            .filter(KnowledgeChunk.id.in_(chunk_ids))
            .all()
        )
        row_map = {c.id: c for c in rows}

        results = []
        for cid in chunk_ids:
            c = row_map.get(cid)
            if c is None:
                continue
            sim = sim_map.get(cid)
            if sim is not None and sim < min_similarity:
                continue
            results.append(RetrievedChunk(
                chunk_id=c.id,
                document_id=c.document_id,
                course_name=c.document.course_name if c.document else "",
                title=c.document.title if c.document else "",
                text=c.text,
                chunk_index=c.chunk_index,
                topic_id=c.topic_id,
                similarity=sim,
            ))
        return results

    def close(self) -> None:
        self._weaviate.close()

    def _retrieve_random(self, topic_id: int, top_k: int) -> list[RetrievedChunk]:
        rows = (
            self.db.query(KnowledgeChunk)
            .filter(KnowledgeChunk.topic_id == topic_id)
            .order_by(text("RANDOM()"))
            .limit(top_k)
            .all()
        )
        return [
            RetrievedChunk(
                chunk_id=c.id,
                document_id=c.document_id,
                course_name=c.document.course_name,
                title=c.document.title,
                text=c.text,
                chunk_index=c.chunk_index,
                topic_id=c.topic_id,
                similarity=None,
            )
            for c in rows
        ]

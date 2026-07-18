from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import tiktoken
from pypdf import PdfReader
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models import KnowledgeChunk, KnowledgeDocument, Topic
from app.models.rag import GeneratedQuestionEvidence
from app.services.llm_client import LLMClient
from app.services.weaviate_client import WeaviateService

log = get_logger(__name__)

_ENC = tiktoken.get_encoding("cl100k_base")

SUPPORTED_EXTENSIONS = {".pdf"}

# Explicit mapping: course folder name → Phase II topic name
# Add new course folders here instead of relying on fuzzy string normalization.
COURSE_TOPIC_NAMES: dict[str, str] = {
    "big data knowledge discovery": "Big Data & Knowledge Discovery",
    "cloud computing": "Cloud Computing",
    "cyber security": "Cybersecurity",
    "deep learning & neural networks": "Deep Learning & Neural Networks",
    "ethics & legal aspects of ai": "Ethics & Legal Aspects of AI",
    "expert systems & knowledge bases": "Expert Systems & Knowledge Bases",
    "graduation thesis": "Graduation Thesis",
    "internet of things": "Internet of Things",
    "machine learning": "Machine Learning",
    "natural language processing": "Natural Language Processing",
    "recurrent & reinforcement learning": "Recurrent & Reinforcement Learning",
}


@dataclass
class IngestResult:
    documents_created: int = 0
    chunks_created: int = 0
    chunks_embedded: int = 0
    errors: list[str] = None
    quarantined: list[str] = None
    zero_chunk_docs: list[int] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.quarantined is None:
            self.quarantined = []
        if self.zero_chunk_docs is None:
            self.zero_chunk_docs = []


@dataclass
class _Source:
    course_name: str
    title: str
    source_path: str
    resource_type: str


@dataclass
class _Chunk:
    source: _Source
    chunk_index: int
    text: str


class RAGIngestionService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()
        self._topic_cache: dict[str, int] = {}
        self._load_topic_cache()

        self._llm = LLMClient(provider=self.settings.RAG_EMBEDDING_PROVIDER)
        self._weaviate = WeaviateService()
        if not self._llm.is_available:
            log.warning("No LLM client available — ingestion will skip embedding generation")

    # ── Public API — Combined (Phase 1 + Phase 2) ───────────────

    def ingest_all(self, force: bool = False) -> IngestResult:
        result = IngestResult()
        sources = list(self._discover_sources())
        if not sources:
            result.errors.append("No course sources found — check RAG_SOURCE_ROOT path")
            return result

        try:
            self._validate_course_mappings({s.course_name for s in sources})
        except ValueError as exc:
            result.errors.append(str(exc))
            return result

        course_counts: dict[str, int] = {}
        for s in sources:
            course_counts[s.course_name] = course_counts.get(s.course_name, 0) + 1
        log.info("=" * 60)
        log.info("INGEST START — %d sources across %d courses", len(sources), len(course_counts))
        for course, count in sorted(course_counts.items()):
            log.info("  %s: %d file(s)", course, count)

        for i, src in enumerate(sources, 1):
            chunks_before = result.chunks_created
            emb_before = result.chunks_embedded
            log.info("[%d/%d] %s / %s — extracting, chunking & embedding ...", i, len(sources), src.course_name, src.source_path)
            try:
                doc = self._extract_and_chunk_source(src, result, force=force)
                if doc is not None:
                    self._embed_and_store_document(doc, result)
                    self.db.commit()
                    self.db.refresh(doc)
                    delta_c = result.chunks_created - chunks_before
                    if doc.embedding_status == "completed":
                        log.info("  ✓ %s / %s — %d chunks, %d embedded, committed", src.course_name, src.source_path, delta_c, result.chunks_embedded - emb_before)
                    else:
                        log.warning("  ⚠ %s / %s — extracted (%d chunks) but embed failed", src.course_name, src.source_path, delta_c)
                else:
                    self.db.commit()
                    log.info("  - %s / %s — skipped (already exists or empty)", src.course_name, src.source_path)
            except Exception as exc:
                self.db.rollback()
                msg = f"{src.course_name}/{src.source_path}: {exc}"
                log.error("  ✗ %s / %s — FAILED: %s", src.course_name, src.source_path, exc)
                result.errors.append(msg)

        log.info("-" * 60)
        log.info(
            "INGEST DONE — %d docs, %d chunks (%d embedded), %d error(s)",
            result.documents_created, result.chunks_created,
            result.chunks_embedded, len(result.errors),
        )
        if result.errors:
            for err in result.errors:
                log.warning("  Error: %s", err)
        log.info("=" * 60)
        return result

    def ingest_course(self, course_name: str, force: bool = False) -> IngestResult:
        result = IngestResult()
        sources = [s for s in self._discover_sources() if s.course_name == course_name]
        if not sources:
            result.errors.append(f"Course '{course_name}' not found in source root")
            return result

        try:
            self._validate_course_mappings({course_name})
        except ValueError as exc:
            result.errors.append(str(exc))
            return result

        log.info("=" * 60)
        log.info("INGEST COURSE — %s (%d sources)", course_name, len(sources))
        for i, src in enumerate(sources, 1):
            chunks_before = result.chunks_created
            emb_before = result.chunks_embedded
            log.info("[%d/%d] %s — extracting, chunking & embedding ...", i, len(sources), src.source_path)
            try:
                doc = self._extract_and_chunk_source(src, result, force=force)
                if doc is not None:
                    self._embed_and_store_document(doc, result)
                    self.db.commit()
                    self.db.refresh(doc)
                    delta_c = result.chunks_created - chunks_before
                    if doc.embedding_status == "completed":
                        log.info("  ✓ %s — %d chunks, %d embedded, committed", src.source_path, delta_c, result.chunks_embedded - emb_before)
                    else:
                        log.warning("  ⚠ %s — extracted (%d chunks) but embed failed", src.source_path, delta_c)
                else:
                    self.db.commit()
                    log.info("  - %s — skipped (already exists or empty)", src.source_path)
            except Exception as exc:
                self.db.rollback()
                msg = f"{src.course_name}/{src.source_path}: {exc}"
                log.error("  ✗ %s — FAILED: %s", src.source_path, exc)
                result.errors.append(msg)
        log.info("-" * 60)
        log.info(
            "INGEST DONE — %d docs, %d chunks (%d embedded), %d error(s)",
            result.documents_created, result.chunks_created,
            result.chunks_embedded, len(result.errors),
        )
        log.info("=" * 60)
        return result

    # ── Public API — Phase 1: Extract + Chunk only (no API calls) ─

    def extract_and_chunk_all(self, force: bool = False) -> IngestResult:
        sources = list(self._discover_sources())
        if not sources:
            result = IngestResult()
            result.errors.append("No course sources found — check RAG_SOURCE_ROOT path")
            return result
        try:
            self._validate_course_mappings({s.course_name for s in sources})
        except ValueError as exc:
            result = IngestResult()
            result.errors.append(str(exc))
            return result
        return self._run_extract_phase(sources, force=force)

    def extract_and_chunk_course(self, course_name: str, force: bool = False) -> IngestResult:
        sources = [s for s in self._discover_sources() if s.course_name == course_name]
        if not sources:
            result = IngestResult()
            result.errors.append(f"Course '{course_name}' not found in source root")
            return result
        try:
            self._validate_course_mappings({course_name})
        except ValueError as exc:
            result = IngestResult()
            result.errors.append(str(exc))
            return result
        return self._run_extract_phase(sources, course_name, force=force)

    def _run_extract_phase(self, sources: list[_Source], course_name: str | None = None, force: bool = False) -> IngestResult:
        result = IngestResult()
        if not sources:
            result.errors.append("No course sources found — check RAG_SOURCE_ROOT path")
            return result

        label = f"course '{course_name}'" if course_name else "all courses"
        log.info("=" * 60)
        log.info("EXTRACT PHASE — %s (%d sources)", label, len(sources))
        for i, src in enumerate(sources, 1):
            log.info("[%d/%d] %s / %s — extracting & chunking ...", i, len(sources), src.course_name, src.source_path)
            try:
                self._extract_and_chunk_source(src, result, force=force)
                self.db.commit()
            except Exception as exc:
                self.db.rollback()
                msg = f"{src.course_name}/{src.source_path}: {exc}"
                log.error("  ✗ %s — FAILED: %s", src.source_path, exc)
                result.errors.append(msg)
        log.info("-" * 60)
        log.info(
            "EXTRACT DONE — %d docs, %d chunks, %d error(s)",
            result.documents_created, result.chunks_created, len(result.errors),
        )
        log.info("=" * 60)
        return result

    # ── Public API — Phase 2: Embed + Store (API calls) ─────────

    def embed_all_pending(self) -> IngestResult:
        result = IngestResult()
        docs = (
            self.db.query(KnowledgeDocument)
            .filter(KnowledgeDocument.embedding_status.in_(["pending", "failed"]))
            .all()
        )
        if not docs:
            result.errors.append("No pending documents found — run extract first")
            return result
        log.info("=" * 60)
        log.info("EMBED PHASE — %d pending/failed documents", len(docs))
        return self._run_embed_phase(docs, result)

    def embed_pending_for_course(self, course_name: str) -> IngestResult:
        result = IngestResult()
        docs = (
            self.db.query(KnowledgeDocument)
            .filter(
                KnowledgeDocument.course_name == course_name,
                KnowledgeDocument.embedding_status.in_(["pending", "failed"]),
            )
            .all()
        )
        if not docs:
            result.errors.append(f"No pending documents for course '{course_name}'")
            return result
        log.info("=" * 60)
        log.info("EMBED PHASE — course '%s' (%d pending/failed documents)", course_name, len(docs))
        return self._run_embed_phase(docs, result)

    def _run_embed_phase(self, docs: list[KnowledgeDocument], result: IngestResult) -> IngestResult:
        for i, doc in enumerate(docs, 1):
            log.info("[%d/%d] doc %d — %s / %s", i, len(docs), doc.id, doc.course_name, doc.title)
            try:
                self._embed_and_store_document(doc, result)
                self.db.commit()
                self.db.refresh(doc)
                if doc.embedding_status == "completed":
                    log.info("  ✓ doc %d — completed (%d chunks)", doc.id, result.chunks_embedded)
                else:
                    log.warning("  ⚠ doc %d — %s", doc.id, doc.embedding_status)
            except Exception as exc:
                self.db.rollback()
                doc.embedding_status = "failed"
                self.db.commit()
                msg = f"doc {doc.id} ({doc.course_name}/{doc.title}): {exc}"
                log.error("  ✗ %s", msg)
                result.errors.append(msg)
        log.info("-" * 60)
        log.info(
            "EMBED DONE — %d documents, %d chunks embedded, %d error(s)",
            len(docs), result.chunks_embedded, len(result.errors),
        )
        log.info("=" * 60)
        return result

    # ── Discovery ───────────────────────────────────────────────

    def _discover_sources(self) -> Iterator[_Source]:
        root = Path(self.settings.RAG_SOURCE_ROOT)
        if not root.is_dir():
            log.warning("RAG source root %s does not exist", root)
            return

        for course_dir in sorted(root.iterdir()):
            if not course_dir.is_dir():
                continue
            course_name = course_dir.name
            for extracted in self._scan_course_dir(course_dir, course_name):
                yield extracted

    def _scan_course_dir(self, course_dir: Path, course_name: str) -> Iterator[_Source]:
        for pdf_path in sorted(course_dir.rglob("*")):
            if pdf_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            title = pdf_path.stem.replace("_", " ").replace("-", " ").strip()
            rel = pdf_path.relative_to(course_dir)
            yield _Source(
                course_name=course_name,
                title=title,
                source_path=str(rel.as_posix()),
                resource_type="pdf",
            )

    # ── Phase 1: Extract + Chunk ─────────────────────────────────

    def _extract_and_chunk_source(self, src: _Source, result: IngestResult, force: bool = False) -> KnowledgeDocument | None:
        """Extract text from PDF, chunk it, save to PG with embedding_status='pending'.
        When force=True, deletes and re-extracts even if a document already exists for this source.
        Returns the KnowledgeDocument or None if skipped/failed."""
        full_path = Path(self.settings.RAG_SOURCE_ROOT) / src.course_name / src.source_path
        if not full_path.is_file():
            result.errors.append(f"File not found: {full_path}")
            return None

        existing = self.db.query(KnowledgeDocument).filter(
            KnowledgeDocument.source_path == src.source_path,
            KnowledgeDocument.course_name == src.course_name,
        ).first()
        if existing:
            if force:
                log.info("Force re-ingesting %s/%s — removing existing doc %d", src.course_name, src.source_path, existing.id)
                # Cascade-delete evidence rows first (FK RESTRICT on chunk_id would block deletion)
                old_chunk_ids = [c.id for c in existing.chunks]
                if old_chunk_ids:
                    deleted = (
                        self.db.query(GeneratedQuestionEvidence)
                        .filter(GeneratedQuestionEvidence.chunk_id.in_(old_chunk_ids))
                        .delete(synchronize_session="fetch")
                    )
                    if deleted:
                        log.info("Cascade-deleted %d GeneratedQuestionEvidence rows for doc %d", deleted, existing.id)
                self._weaviate.delete_document_chunks(existing.id)
                self.db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == existing.id).delete()
                self.db.delete(existing)
                self.db.flush()
            else:
                log.info("Skipping already-extracted %s/%s", src.course_name, src.source_path)
                return None

        raw = self._extract_text(full_path)
        log.info("Extracted %d chars from %s", len(raw), src.source_path)

        # OCR / empty-PDF quarantine
        text_stripped = raw.strip()
        if not text_stripped:
            msg = f"Empty or unreadable (no extractable text): {full_path}"
            if self.settings.RAG_OCR_ENABLED:
                result.quarantined.append(f"{full_path} — no text, OCR configured but not yet implemented")
            else:
                result.quarantined.append(f"{full_path} — no text, consider enabling RAG_OCR_ENABLED or check PDF")
            result.errors.append(msg)
            return None
        if len(text_stripped) < 50:
            log.warning("Very short text (%d chars) from %s — may be scanned or corrupted", len(text_stripped), src.source_path)

        topic_id = self._resolve_topic(src.course_name)
        if topic_id is None:
            result.errors.append(f"Course '{src.course_name}' has no topic mapping — add to COURSE_TOPIC_NAMES")
            return None
        chunks = self._chunk_text(raw, src)
        total = len(chunks)
        log.info("Chunked into %d chunks (size=%d, overlap=%d)", total, self.settings.RAG_CHUNK_SIZE, self.settings.RAG_CHUNK_OVERLAP)

        doc = KnowledgeDocument(
            course_name=src.course_name,
            title=src.title,
            topic_id=topic_id,
            source_path=src.source_path,
            resource_type=src.resource_type,
            embedding_status="pending",
        )
        self.db.add(doc)
        self.db.flush()
        result.documents_created += 1

        for c in chunks:
            kc = KnowledgeChunk(
                document_id=doc.id,
                chunk_index=c.chunk_index,
                text=c.text,
                topic_id=topic_id,
            )
            self.db.add(kc)
        self.db.flush()
        result.chunks_created += total

        log.info("Saved %s — %d chunks to PG (pending)", src.source_path, total)
        return doc

    # ── Phase 2: Embed + Store ──────────────────────────────────

    def _embed_and_store_document(self, doc: KnowledgeDocument, result: IngestResult) -> None:
        """Batch-embed all chunks for a document, store in Weaviate.
        Sets embedding_status to 'completed' only if all chunks stored successfully."""
        if doc.embedding_status == "completed":
            return

        chunks = (
            self.db.query(KnowledgeChunk)
            .filter(KnowledgeChunk.document_id == doc.id)
            .order_by(KnowledgeChunk.chunk_index)
            .all()
        )
        if not chunks:
            log.warning("Document %d has no chunks — marking as failed", doc.id)
            doc.embedding_status = "failed"
            self.db.flush()
            result.errors.append(f"doc {doc.id}: zero chunks — cannot embed")
            return

        # ── Phase 2a: batch-embed ────────────────────────────
        texts = [c.text for c in chunks]
        log.info("Batch-embedding %d chunks via %s ...", len(texts), self.settings.RAG_EMBEDDING_PROVIDER)
        embeddings = self._llm.embed(texts)

        # Filter to successfully embedded texts only
        valid: list[tuple[KnowledgeChunk, list[float]]] = [
            (c, emb) for c, emb in zip(chunks, embeddings) if emb is not None
        ]
        if not valid:
            log.warning("Document %d: no chunks could be embedded — marking failed", doc.id)
            doc.embedding_status = "failed"
            self.db.flush()
            result.chunks_embedded += 0
            result.errors.append(f"doc {doc.id}: embedding returned all None")
            return

        # ── Phase 2b: store in Weaviate ──────────────────────
        self._weaviate.delete_document_chunks(doc.id)

        stored = 0
        for chunk, emb in valid:
            if self._weaviate.store_chunk(
                chunk_id=chunk.id,
                document_id=doc.id,
                course_name=doc.course_name,
                title=doc.title,
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                topic_id=doc.topic_id,
                embedding=emb,
            ):
                stored += 1

        result.chunks_embedded += stored

        if stored == len(valid):
            doc.embedding_status = "completed"
        else:
            doc.embedding_status = "failed"
            msg = f"doc {doc.id}: stored {stored}/{len(valid)} chunks — Weaviate failure"
            log.warning(msg)
            result.errors.append(msg)

        self.db.flush()
        log.info("Document %d: %d/%d chunks embedded, %d stored in Weaviate", doc.id, len(valid), len(texts), stored)

    # ── PDF Extraction ──────────────────────────────────────────

    def _extract_text(self, path: Path) -> str:
        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        raw = "\n\n".join(pages)
        return re.sub(r"[\ud800-\udfff\x00]", "", raw)

    # ── Chunking ────────────────────────────────────────────────

    def _chunk_text(self, text: str, src: _Source) -> list[_Chunk]:
        chunk_size = self.settings.RAG_CHUNK_SIZE
        overlap = self.settings.RAG_CHUNK_OVERLAP

        paragraphs = re.split(r"\n\s*\n", text)
        tokens_per_para = [len(_ENC.encode(p)) for p in paragraphs]

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for para, pt in zip(paragraphs, tokens_per_para):
            if pt > chunk_size:
                if current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_tokens = 0
                for start in range(0, pt, chunk_size - overlap):
                    segment = self._encode_slice(para, start, start + chunk_size)
                    chunks.append(segment)
                continue

            if current_tokens + pt > chunk_size and current:
                chunks.append("\n\n".join(current))
                overlap_text = self._take_overlap(current, overlap)
                current = [overlap_text] if overlap_text else []
                current_tokens = len(_ENC.encode(overlap_text)) if overlap_text else 0

            current.append(para)
            current_tokens += pt

        if current:
            chunks.append("\n\n".join(current))

        return [_Chunk(source=src, chunk_index=i, text=t) for i, t in enumerate(chunks)]

    @staticmethod
    def _encode_slice(text: str, start: int, end: int) -> str:
        tokens = _ENC.encode(text)[start:end]
        return _ENC.decode(tokens)

    @staticmethod
    def _take_overlap(paragraphs: list[str], overlap_tokens: int) -> str:
        parts: list[str] = []
        tokens = 0
        for p in reversed(paragraphs):
            pt = len(_ENC.encode(p))
            if tokens + pt > overlap_tokens:
                remaining = overlap_tokens - tokens
                if remaining > 0:
                    encoded = _ENC.encode(p)
                    parts.insert(0, _ENC.decode(encoded[-remaining:]))
                break
            parts.insert(0, p)
            tokens += pt
        return "\n\n".join(parts)

    # ── Embedding ───────────────────────────────────────────────

    def _embed_one(self, text: str) -> list[float] | None:
        result = self._llm.embed([text])
        return result[0] if result else None

    # ── Reconciliation / Verification ─────────────────────────

    def verify(self) -> dict:
        """Compare PG docs/chunks vs Weaviate stored vectors.
        Returns a structured report with mismatches and sample searches per topic."""
        report: dict = {
            "pg_documents": 0,
            "pg_chunks": 0,
            "weaviate_objects": 0,
            "objects_orphaned": 0,
            "chunks_missing": [],
            "topic_coverage": {},
            "sample_searches": [],
            "errors": [],
        }

        # PG counts
        report["pg_documents"] = self.db.query(KnowledgeDocument).count()
        report["pg_chunks"] = self.db.query(KnowledgeChunk).count()

        # Weaviate count + per-topic breakdown
        if not self._weaviate.is_available:
            report["errors"].append("Weaviate not reachable — cannot verify vector store")
            return report

        try:
            wv_count, orphaned = self._weaviate.count_objects()
            report["weaviate_objects"] = wv_count
            report["objects_orphaned"] = orphaned
        except Exception as exc:
            report["errors"].append(f"Failed to count Weaviate objects: {exc}")

        # Chunks in PG but not in Weaviate (by trying to delete-check or by diff)
        completed_docs = (
            self.db.query(KnowledgeDocument)
            .filter(KnowledgeDocument.embedding_status == "completed")
            .all()
        )
        for doc in completed_docs:
            chunk_ids = [
                c.id
                for c in self.db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == doc.id).all()
            ]
            for cid in chunk_ids:
                if not self._weaviate.chunk_exists(cid):
                    report["chunks_missing"].append(cid)

        # Topic coverage
        topics = self.db.query(Topic).filter(
            Topic.phase_id == self.settings.PHASE2_PHASE_ID
        ).all()
        for topic in topics:
            count = (
                self.db.query(KnowledgeChunk)
                .filter(KnowledgeChunk.topic_id == topic.id)
                .count()
            )
            if count > 0:
                report["topic_coverage"][topic.name] = count

        # Sample retrieval per topic (vector search smoke test)
        if self._llm.is_available and self._weaviate.is_available:
            topics_with_chunks = [
                t for t in topics if report["topic_coverage"].get(t.name, 0) > 0
            ]
            for topic in topics_with_chunks[:3]:  # limit to 3 topics to avoid rate limits
                try:
                    emb = self._llm.embed([f"sample query for {topic.name}"])
                    if emb and emb[0]:
                        results = self._weaviate.search(
                            query_embedding=emb[0],
                            topic_id=topic.id,
                            top_k=3,
                        )
                        report["sample_searches"].append({
                            "topic_id": topic.id,
                            "topic_name": topic.name,
                            "results_found": len(results),
                            "similarities": [r.get("similarity") for r in results[:3] if r.get("similarity")],
                        })
                except Exception as exc:
                    report["sample_searches"].append({
                        "topic_id": topic.id,
                        "topic_name": topic.name,
                        "error": str(exc),
                    })

        return report

    def close(self) -> None:
        self._weaviate.close()

    # ── Readiness Check ──────────────────────────────────────

    def readiness_check(self, smoke_test: bool = False) -> dict:
        """Validate all pre-conditions for ingestion.
        Returns {'status': 'ok'|'fail', 'issues': [...], ...}.
        When smoke_test=True, runs a live embed+search against the first available topic."""
        issues: list[str] = []
        info: dict = {}

        # 1. API key
        if not self._llm.is_available:
            issues.append(f"LLM client unavailable (provider={self.settings.RAG_EMBEDDING_PROVIDER}) — check API key")

        # 2. Weaviate
        weaviate_ok = self._weaviate.is_available
        if not weaviate_ok:
            issues.append("Weaviate not reachable — check docker compose ps")
        info["weaviate_ok"] = weaviate_ok

        # 3. Source root
        root = Path(self.settings.RAG_SOURCE_ROOT)
        if not root.is_dir():
            issues.append(f"RAG_SOURCE_ROOT '{root}' does not exist")
        else:
            courses = sorted(d.name for d in root.iterdir() if d.is_dir())
            info["course_dirs"] = courses
            if not courses:
                issues.append(f"No course directories found in {root}")
            else:
                unmapped = [c for c in courses if self._resolve_topic(c) is None]
                if unmapped:
                    issues.append(f"Unmapped courses: {', '.join(unmapped)}")

                # 4. Verify Phase II topics exist for mapped courses
                missing_topics: list[str] = []
                for c in courses:
                    tid = self._resolve_topic(c)
                    if tid is None:
                        continue
                    topic = self.db.query(Topic).filter(Topic.id == tid).first()
                    if topic is None:
                        missing_topics.append(f"{c} → topic_id={tid} not found in DB")
                if missing_topics:
                    issues.append(f"Phase II topic(s) missing: {'; '.join(missing_topics)}")

        # 5. Zero-chunk documents
        zero_chunk_docs = (
            self.db.query(KnowledgeDocument)
            .filter(KnowledgeDocument.embedding_status == "failed")
            .all()
        )
        zero_chunk_report = [
            f"doc {d.id} ({d.course_name}/{d.title}) — {d.embedding_status}"
            for d in zero_chunk_docs
            if not self.db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == d.id).first()
        ]
        info["zero_chunk_docs"] = zero_chunk_report

        # 6. Embedding stats
        total_docs = self.db.query(KnowledgeDocument).count()
        statuses = ["pending", "completed", "failed"]
        info["doc_counts"] = {
            s: self.db.query(KnowledgeDocument).filter(KnowledgeDocument.embedding_status == s).count()
            for s in statuses
        }
        info["total_docs"] = total_docs

        # 7. Optional smoke test (live embed + search against first topic)
        if smoke_test and weaviate_ok and self._llm.is_available:
            try:
                first_topic = self.db.query(Topic).filter(
                    Topic.phase_id == self.settings.PHASE2_PHASE_ID
                ).first()
                if first_topic:
                    emb = self._llm.embed(["smoke test query"])
                    if emb and emb[0]:
                        results = self._weaviate.search(
                            query_embedding=emb[0],
                            topic_id=first_topic.id,
                            top_k=3,
                        )
                        info["smoke_test"] = {
                            "topic_id": first_topic.id,
                            "topic_name": first_topic.name,
                            "results_count": len(results),
                        }
                    else:
                        info["smoke_test"] = {"error": "embedding failed"}
                else:
                    info["smoke_test"] = {"error": "no Phase II topics found"}
            except Exception as exc:
                info["smoke_test"] = {"error": str(exc)}

        if issues:
            return {"status": "fail", "issues": issues, **info}
        return {"status": "ok", "issues": [], **info}

    # ── Topic Mapping —────────────────────────────────────────

    def _load_topic_cache(self) -> None:
        """Load Phase II topics: topic_name → topic_id."""
        topics = (
            self.db.query(Topic)
            .filter(Topic.phase_id == self.settings.PHASE2_PHASE_ID)
            .all()
        )
        self._topic_cache = {t.name: t.id for t in topics}

    def _resolve_topic(self, course_name: str) -> int | None:
        """Look up topic_id from the explicit COURSE_TOPIC_NAMES mapping."""
        topic_name = COURSE_TOPIC_NAMES.get(course_name.strip().lower())
        if topic_name is None:
            return None
        return self._topic_cache.get(topic_name)

    def _validate_course_mappings(self, course_names: set[str]) -> None:
        """Raise ValueError if any course folder has no topic mapping."""
        unmapped = sorted(c for c in course_names if self._resolve_topic(c) is None)
        if unmapped:
            msg = (
                f"Unmapped course(s): {', '.join(unmapped)}. "
                f"Add an entry to COURSE_TOPIC_NAMES in {__name__} "
                f"and ensure the topic exists with phase_id={self.settings.PHASE2_PHASE_ID}."
            )
            raise ValueError(msg)

from __future__ import annotations

from dataclasses import dataclass

from app.core.logging import get_logger
from app.services.rag_retrieval_service import RetrievedChunk

log = get_logger(__name__)


@dataclass
class EvidenceReport:
    """Result of evidence sufficiency check."""

    sufficient: bool
    reason: str
    chunk_count: int
    avg_similarity: float
    high_quality_count: int  # chunks with similarity >= HIGH_SIM_THRESHOLD

    HIGH_SIM_THRESHOLD: float = 0.5
    MIN_CHUNKS: int = 1
    MIN_AVG_SIMILARITY: float = 0.15


class EvidenceValidationService:
    """Deterministic evidence sufficiency gate.

    Checks whether the compressed chunks provide enough signal for
    quality question generation.  No LLM call.

    Sufficiency criteria (all must pass):
      1. At least ``MIN_CHUNKS`` chunks remain.
      2. Average similarity across chunks >= ``MIN_AVG_SIMILARITY``.
      3. At least one chunk has similarity >= ``HIGH_SIM_THRESHOLD``
         (ensures at least one strongly relevant chunk).

    If any criterion fails, ``sufficient=False`` with a human-readable
    reason so the graph can abort cleanly.
    """

    MIN_CHUNKS: int = 1
    MIN_AVG_SIMILARITY: float = 0.15
    HIGH_SIM_THRESHOLD: float = 0.5

    def validate(
        self,
        chunks: list[RetrievedChunk],
        *,
        min_chunks: int | None = None,
        min_avg_similarity: float | None = None,
        high_sim_threshold: float | None = None,
    ) -> EvidenceReport:
        """Return an EvidenceReport describing whether evidence is sufficient."""
        min_chunks = min_chunks if min_chunks is not None else self.MIN_CHUNKS
        min_avg_similarity = (
            min_avg_similarity
            if min_avg_similarity is not None
            else self.MIN_AVG_SIMILARITY
        )
        high_sim_threshold = (
            high_sim_threshold
            if high_sim_threshold is not None
            else self.HIGH_SIM_THRESHOLD
        )

        chunk_count = len(chunks)

        # Criterion 1: minimum chunk count
        if chunk_count < min_chunks:
            return EvidenceReport(
                sufficient=False,
                reason=f"Too few chunks: {chunk_count} < {min_chunks}",
                chunk_count=chunk_count,
                avg_similarity=0.0,
                high_quality_count=0,
            )

        # Compute similarity stats
        sims = [c.similarity or 0.0 for c in chunks]
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        high_quality = sum(1 for s in sims if s >= high_sim_threshold)

        # Criterion 2: average similarity floor
        if avg_sim < min_avg_similarity:
            return EvidenceReport(
                sufficient=False,
                reason=f"Average similarity too low: {avg_sim:.3f} < {min_avg_similarity}",
                chunk_count=chunk_count,
                avg_similarity=avg_sim,
                high_quality_count=high_quality,
            )

        # Criterion 3: at least one high-quality chunk
        if high_quality < 1:
            return EvidenceReport(
                sufficient=False,
                reason=f"No high-quality chunks (sim >= {high_sim_threshold})",
                chunk_count=chunk_count,
                avg_similarity=avg_sim,
                high_quality_count=high_quality,
            )

        return EvidenceReport(
            sufficient=True,
            reason="Evidence is sufficient",
            chunk_count=chunk_count,
            avg_similarity=avg_sim,
            high_quality_count=high_quality,
        )

from __future__ import annotations

from app.core.logging import get_logger

log = get_logger(__name__)


class RetrievalRepairService:
    """Generate broader, simpler queries when evidence is insufficient.

    Deterministic, no LLM call.  Bounded by graph-level retry count
    (``MAX_RETRIES`` in the workflow module).

    Strategy: propose queries that are strictly simpler/broader than
    the difficulty-band-adjusted queries the planner generates, then
    exclude any query already tried.  This casts a wider net in the
    vector store to pick up chunks the original pass missed.
    """

    def repair(
        self,
        topic_name: str,
        original_queries: list[str],
    ) -> list[str]:
        """Return a new set of broader candidate queries.

        Parameters
        ----------
        topic_name:
            Canonical topic name (e.g. ``"Cryptography"``).
        original_queries:
            Queries already tried in the first retrieval pass.

        Returns
        -------
        Non-empty list of queries that were **not** in *original_queries*.
        If every candidate is a duplicate, falls back to bare topic name.
        """
        seen = {q.strip().lower() for q in original_queries if q}

        candidates = [
            topic_name,
            f"basic fundamentals of {topic_name}",
            f"key principles of {topic_name}",
        ]

        deduped: list[str] = []
        for q in candidates:
            normalised = q.strip().lower()
            if normalised not in seen:
                seen.add(normalised)
                deduped.append(q.strip())

        if not deduped:
            deduped = [topic_name]

        log.debug(
            "RetrievalRepair: %d new queries from topic=%r (skipped %d original)",
            len(deduped),
            topic_name,
            len(original_queries),
        )
        return deduped

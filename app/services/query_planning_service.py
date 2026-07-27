from __future__ import annotations

from dataclasses import dataclass, field

from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class QueryPlan:
    """Output of the query planner — a deduplicated list of retrieval queries."""

    queries: list[str]
    primary_query: str


class QueryPlannerService:
    """Deterministic, template-based query planner for RAG retrieval.

    Generates a small set of candidate retrieval queries based on:
    - topic name
    - target difficulty (theta)
    - likely misconception / remediation angle
    - simple paraphrase / synonym variant

    No LLM call is made — this is purely template-driven so it is
    fast, deterministic, and failure-safe.  If the planner ever returns
    an empty list, the caller falls back to the legacy single-query path.
    """

    MAX_QUERIES: int = 4

    # Difficulty-band labels derived from theta
    _BANDS: list[tuple[float, str]] = [
        (1.5, "expert-level edge cases in"),
        (0.5, "advanced concepts in"),
        (-0.5, ""),
        (float("-inf"), "basic fundamentals of"),
    ]

    # Misconception / remediation angles — appended to the topic
    _MISCONCEPTION_ANGLES: list[str] = [
        "common mistakes in",
        "frequently confused concepts in",
    ]

    # Paraphrase templates — wraps the topic name
    _PARAPHRASE_TEMPLATES: list[str] = [
        "{topic}",
        "key principles of {topic}",
    ]

    def plan(
        self,
        topic_name: str,
        theta: float,
        max_queries: int | None = None,
    ) -> QueryPlan:
        """Return a deduplicated list of candidate retrieval queries.

        Parameters
        ----------
        topic_name:
            Canonical topic name (e.g. "Cryptography").
        theta:
            Student ability estimate (IRT scale).
        max_queries:
            Cap on output queries. Defaults to ``MAX_QUERIES``.

        Returns
        -------
        QueryPlan with ``primary_query`` (the best single query, always
        present) and ``queries`` (deduplicated list, always includes
        primary).
        """
        max_queries = max_queries or self.MAX_QUERIES

        primary = self._primary_query(topic_name, theta)
        candidates: list[str] = [primary]

        # Misconception angle
        angle = self._misconception_angle(topic_name)
        candidates.append(angle)

        # Paraphrase variants
        for tmpl in self._PARAPHRASE_TEMPLATES:
            q = tmpl.format(topic=topic_name)
            candidates.append(q)

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for q in candidates:
            normalised = q.strip().lower()
            if normalised not in seen:
                seen.add(normalised)
                deduped.append(q.strip())

        # Respect cap — primary always kept
        if len(deduped) > max_queries:
            deduped = [primary] + [
                q for q in deduped[1:] if q != primary
            ][: max_queries - 1]

        log.debug("QueryPlanner: %d queries for topic=%r theta=%.2f", len(deduped), topic_name, theta)
        return QueryPlan(queries=deduped, primary_query=primary)

    # ── helpers ────────────────────────────────────────────────

    def _primary_query(self, topic_name: str, theta: float) -> str:
        band_label = self._difficulty_band(theta)
        if band_label:
            return f"{band_label} {topic_name}"
        return topic_name

    def _difficulty_band(self, theta: float) -> str:
        for threshold, label in self._BANDS:
            if theta >= threshold:
                return label
        return ""

    def _misconception_angle(self, topic_name: str) -> str:
        return f"{self._MISCONCEPTION_ANGLES[0]} {topic_name}"

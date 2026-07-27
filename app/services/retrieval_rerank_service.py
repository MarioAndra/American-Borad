from __future__ import annotations

import re
from collections.abc import Iterable

from app.core.logging import get_logger
from app.services.rag_retrieval_service import RetrievedChunk

log = get_logger(__name__)

_WORD_RE = re.compile(r"\b[a-z0-9]{2,}\b")


def _tokenize(text: str) -> set[str]:
    """Lowercased word tokens, stripped of short / stopword-like noise."""
    return {t for t in _WORD_RE.findall(text.lower())}


# Minimal English stopwords — just enough to stop noise like "is", "of".
_STOP: frozenset[str] = frozenset({
    "the", "is", "of", "in", "to", "and", "or", "a", "an", "for",
    "on", "at", "by", "it", "as", "be", "are", "was", "were", "this",
    "that", "with", "from", "can", "may", "not", "but", "if", "so",
})


def _meaningful_tokens(text: str) -> set[str]:
    return _tokenize(text) - _STOP


def lexical_overlap(query: str, chunk_text: str) -> float:
    """Jaccard-like overlap between query tokens and chunk tokens.

    Returns a float in [0, 1].  Stopwords are excluded before
    scoring so the signal reflects actual content overlap.
    """
    q_tokens = _meaningful_tokens(query)
    c_tokens = _meaningful_tokens(chunk_text)
    if not q_tokens:
        return 0.0
    intersection = q_tokens & c_tokens
    union = q_tokens | c_tokens
    return len(intersection) / len(union) if union else 0.0


class RerankerService:
    """Deterministic lexical + similarity fusion reranker.

    For each chunk the final score is:

        score = w_sim * similarity + w_lex * max_lexical_overlap

    where ``max_lexical_overlap`` is the best overlap across all
    candidate queries.  This gives a lightweight, reproducible
    ranking without any LLM call.

    If anything goes wrong the original list is returned unchanged.
    """

    W_SIM: float = 0.6
    W_LEX: float = 0.4

    def rerank(
        self,
        chunks: list[RetrievedChunk],
        queries: list[str],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return chunks reordered by fused score, capped at *top_k*.

        Falls back silently to the original order on any error.
        """
        if not chunks:
            return []

        try:
            return self._rerank(chunks, queries, top_k)
        except Exception:
            log.exception("Reranker failed — returning original chunk order")
            return list(chunks)

    # ------------------------------------------------------------------

    def _rerank(
        self,
        chunks: list[RetrievedChunk],
        queries: list[str],
        top_k: int | None,
    ) -> list[RetrievedChunk]:
        valid_queries = [q for q in queries if q and q.strip()]
        if not valid_queries:
            return list(chunks)

        scored: list[tuple[float, int, RetrievedChunk]] = []
        for idx, chunk in enumerate(chunks):
            sim = chunk.similarity if chunk.similarity is not None else 0.0
            best_lex = max(
                (lexical_overlap(q, chunk.text) for q in valid_queries),
                default=0.0,
            )
            final_score = self.W_SIM * sim + self.W_LEX * best_lex
            scored.append((final_score, idx, chunk))

        scored.sort(key=lambda t: (-t[0], t[1]))  # score desc, stable by original pos
        result = [chunk for _, _, chunk in scored]

        if top_k is not None:
            result = result[:top_k]

        return result

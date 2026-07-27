from __future__ import annotations

from app.core.logging import get_logger
from app.services.rag_retrieval_service import RetrievedChunk

log = get_logger(__name__)


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens, 2+ chars."""
    return {w for w in text.lower().split() if len(w) >= 2}


def text_similarity(a: str, b: str) -> float:
    """Jaccard similarity between two text strings."""
    sa, sb = _tokenize(a), _tokenize(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class ContextCompressionService:
    """Deduplicate and prune low-value chunks after reranking.

    Strategy (deterministic, no LLM call):
      1. Remove near-duplicate chunks (Jaccard >= threshold).
         When two chunks duplicate, keep the one with higher similarity.
      2. Drop chunks whose similarity is below a floor.
      3. Cap the result at ``max_chunks``.

    Falls back silently to returning the original list on any error.
    """

    DEDUP_THRESHOLD: float = 0.85
    MIN_SIMILARITY: float = 0.05
    MAX_CHUNKS: int = 5

    def compress(
        self,
        chunks: list[RetrievedChunk],
        *,
        dedup_threshold: float | None = None,
        min_similarity: float | None = None,
        max_chunks: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return deduplicated, pruned chunks."""
        if not chunks:
            return []

        try:
            return self._compress(
                chunks,
                dedup_threshold=dedup_threshold or self.DEDUP_THRESHOLD,
                min_similarity=min_similarity if min_similarity is not None else self.MIN_SIMILARITY,
                max_chunks=max_chunks or self.MAX_CHUNKS,
            )
        except Exception:
            log.exception("ContextCompression failed — returning original chunks")
            return list(chunks)

    # ------------------------------------------------------------------

    def _compress(
        self,
        chunks: list[RetrievedChunk],
        *,
        dedup_threshold: float,
        min_similarity: float,
        max_chunks: int,
    ) -> list[RetrievedChunk]:
        # 1. Deduplicate — keep chunk with higher similarity on collision
        kept: list[RetrievedChunk] = []
        kept_tokens: list[set[str]] = []

        for chunk in chunks:
            c_tokens = _tokenize(chunk.text)
            is_dup = False

            for i, k_tokens in enumerate(kept_tokens):
                if not k_tokens and not c_tokens:
                    sim = 1.0
                elif not k_tokens or not c_tokens:
                    sim = 0.0
                else:
                    sim = len(c_tokens & k_tokens) / len(c_tokens | k_tokens)

                if sim >= dedup_threshold:
                    # Keep the chunk with higher similarity score
                    existing_sim = kept[i].similarity or 0.0
                    new_sim = chunk.similarity or 0.0
                    if new_sim > existing_sim:
                        kept[i] = chunk
                        kept_tokens[i] = c_tokens
                    is_dup = True
                    break

            if not is_dup:
                kept.append(chunk)
                kept_tokens.append(c_tokens)

        # 2. Drop low-similarity chunks
        if min_similarity > 0:
            kept = [c for c in kept if (c.similarity or 0.0) >= min_similarity]

        # 3. Cap at max_chunks (already reranked, so just truncate)
        result = kept[:max_chunks]

        log.info(
            "ContextCompression: %d chunks -> %d after dedup/prune/cap",
            len(chunks),
            len(result),
        )
        return result

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Text normalisation (deterministic, no LLM)
# ---------------------------------------------------------------------------

_STRIP_RE = re.compile(r"[^a-z0-9\s]")


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, remove punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = _STRIP_RE.sub(" ", text)
    return " ".join(text.split())


def _tokenize(text: str) -> set[str]:
    """Word tokens from normalised text, 2+ chars."""
    return {w for w in normalize_text(text).split() if len(w) >= 2}


def jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity between two text strings."""
    sa, sb = _tokenize(a), _tokenize(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class DedupReport:
    """Result of a duplicate-check against existing questions."""

    is_duplicate: bool
    max_similarity: float
    source: str  # "generated" | "bank" | "none"
    compared_count: int
    threshold: float


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class QuestionDedupService:
    """Deterministic duplicate detection before question persistence.

    Compares generated question text against:
      1. Existing ``GeneratedQuestion`` rows for the same topic.
      2. Fixed question bank (``Question`` model) for the same topic scope.

    Strategy (no LLM call):
      - Normalise + Jaccard similarity against each candidate.
      - Flag as duplicate when max similarity >= threshold.

    Thresholds are explicit class constants, overridable via kwargs.
    """

    GENERATED_DUP_THRESHOLD: float = 0.65
    BANK_DUP_THRESHOLD: float = 0.70

    def check(
        self,
        db: Session,
        *,
        topic_id: int,
        question_text: str,
        generated_threshold: float | None = None,
        bank_threshold: float | None = None,
    ) -> DedupReport:
        """Return a DedupReport describing whether the question is a duplicate.

        Compares against generated questions first (same topic), then the
        fixed question bank.  Returns on the *first* duplicate hit.
        """
        gen_thresh = (
            generated_threshold
            if generated_threshold is not None
            else self.GENERATED_DUP_THRESHOLD
        )
        bank_thresh = (
            bank_threshold
            if bank_threshold is not None
            else self.BANK_DUP_THRESHOLD
        )

        # --- 1. Check against GeneratedQuestion rows for same topic ---
        report = self._check_generated(
            db, topic_id=topic_id, question_text=question_text,
            threshold=gen_thresh,
        )
        if report.is_duplicate:
            return report

        # --- 2. Check against fixed question bank for same topic scope ---
        report = self._check_bank(
            db, topic_id=topic_id, question_text=question_text,
            threshold=bank_thresh,
        )
        return report

    # ------------------------------------------------------------------

    def _check_generated(
        self,
        db: Session,
        *,
        topic_id: int,
        question_text: str,
        threshold: float,
    ) -> DedupReport:
        """Compare against existing GeneratedQuestion rows for the topic."""
        from app.models.rag import GeneratedQuestion

        existing = (
            db.query(GeneratedQuestion.text)
            .filter(GeneratedQuestion.topic_id == topic_id)
            .all()
        )

        if not existing:
            return DedupReport(
                is_duplicate=False,
                max_similarity=0.0,
                source="none",
                compared_count=0,
                threshold=threshold,
            )

        max_sim = 0.0
        for (text,) in existing:
            sim = jaccard_similarity(question_text, text)
            if sim > max_sim:
                max_sim = sim

        return DedupReport(
            is_duplicate=max_sim >= threshold,
            max_similarity=max_sim,
            source="generated",
            compared_count=len(existing),
            threshold=threshold,
        )

    def _check_bank(
        self,
        db: Session,
        *,
        topic_id: int,
        question_text: str,
        threshold: float,
    ) -> DedupReport:
        """Compare against fixed question bank (Question model) for the topic scope.

        Joins Question -> SubTopic -> Topic to find questions whose topic
        matches the given topic_id.
        """
        from app.models.question import Question
        from app.models.subtopic import SubTopic

        existing = (
            db.query(Question.text)
            .join(SubTopic, SubTopic.id == Question.subtopic_id)
            .filter(SubTopic.topic_id == topic_id)
            .all()
        )

        if not existing:
            return DedupReport(
                is_duplicate=False,
                max_similarity=0.0,
                source="none",
                compared_count=0,
                threshold=threshold,
            )

        max_sim = 0.0
        for (text,) in existing:
            sim = jaccard_similarity(question_text, text)
            if sim > max_sim:
                max_sim = sim

        return DedupReport(
            is_duplicate=max_sim >= threshold,
            max_similarity=max_sim,
            source="bank",
            compared_count=len(existing),
            threshold=threshold,
        )

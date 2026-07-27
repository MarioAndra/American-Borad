from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.services.grounding_validation_service import _STOP_WORDS

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Text normalisation — same approach as QuestionDedupService
# ---------------------------------------------------------------------------

_STRIP_RE = re.compile(r"[^a-z0-9\s]")


def _normalize(text: str) -> str:
    """Lowercase, strip accents, remove punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = _STRIP_RE.sub(" ", text)
    return " ".join(text.split())


def _tokenize(text: str) -> set[str]:
    """Word tokens from normalised text, 2+ chars, stop words excluded."""
    return {w for w in _normalize(text).split() if len(w) >= 2 and w not in _STOP_WORDS}


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity between two text strings."""
    sa, sb = _tokenize(a), _tokenize(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class DistractorReport:
    """Result of distractor quality validation."""

    valid: bool
    distinct_distractors: bool
    separated_from_correct: bool
    meaningful_distractors: bool
    issues: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DistractorValidationService:
    """Deterministic distractor quality check on generated MCQ options.

    Validates that:
      1. Every distractor is non-empty and contains real content words.
      2. Distractors are distinct from each other (no duplicates).
      3. No distractor is too similar to the correct answer.
      4. There are enough distractors (≥ 2 incorrect options).

    All thresholds are explicit class constants. No LLM call.
    Exception-safe: on any unexpected error, returns ``valid=False``
    so the question is blocked (fail-closed).
    """

    # Maximum Jaccard similarity allowed between any distractor pair
    MAX_INTRA_DISTRACTOR_SIM: float = 0.85
    # Maximum Jaccard similarity allowed between a distractor and the correct answer
    MAX_VS_ANSWER_SIM: float = 0.70
    # Minimum number of incorrect options required
    MIN_DISTRACTOR_COUNT: int = 2

    def validate(
        self,
        options: list[dict],
    ) -> DistractorReport:
        """Run deterministic distractor quality checks.

        Parameters
        ----------
        options:
            List of option dicts, each with ``text`` (str) and
            ``is_correct`` (bool).

        Returns
        -------
        DistractorReport
            Structured result with individual flags and issues.
        """
        issues: list[str] = []

        try:
            # --- Extract correct and distractors ---
            correct_text = ""
            distractor_texts: list[str] = []

            for opt in options:
                text = (opt.get("text") or "").strip()
                if opt.get("is_correct"):
                    correct_text = text
                else:
                    distractor_texts.append(text)

            # --- Check minimum distractor count ---
            if len(distractor_texts) < self.MIN_DISTRACTOR_COUNT:
                issues.append(
                    f"Too few distractors: {len(distractor_texts)} "
                    f"< {self.MIN_DISTRACTOR_COUNT}"
                )

            # --- Check meaningful distractors (non-empty, real content) ---
            meaningful = True
            for i, text in enumerate(distractor_texts):
                if not text.strip():
                    meaningful = False
                    issues.append(f"Distractor {i + 1} is empty")
                else:
                    tokens = _tokenize(text)
                    if not tokens:
                        meaningful = False
                        issues.append(
                            f"Distractor {i + 1} has no content words "
                            f"after stop-word removal"
                        )

            # --- Check distractors are distinct from each other ---
            distinct = True
            for i in range(len(distractor_texts)):
                for j in range(i + 1, len(distractor_texts)):
                    sim = _jaccard(distractor_texts[i], distractor_texts[j])
                    if sim >= self.MAX_INTRA_DISTRACTOR_SIM:
                        distinct = False
                        issues.append(
                            f"Distractors {i + 1} and {j + 1} are near-duplicates "
                            f"(Jaccard={sim:.2f})"
                        )

            # --- Check distractors are separated from the correct answer ---
            separated = True
            if correct_text:
                for i, text in enumerate(distractor_texts):
                    sim = _jaccard(text, correct_text)
                    if sim >= self.MAX_VS_ANSWER_SIM:
                        separated = False
                        issues.append(
                            f"Distractor {i + 1} too similar to correct answer "
                            f"(Jaccard={sim:.2f})"
                        )
            else:
                # Missing/empty correct answer is a hard failure — we cannot
                # trust a question whose correct option has no text.
                separated = False
                issues.append("Correct answer text is empty — cannot validate distractor separation")

            valid = meaningful and distinct and separated and len(distractor_texts) >= self.MIN_DISTRACTOR_COUNT

            log.info(
                "DistractorValidation: valid=%s, distinct=%s, separated=%s, "
                "meaningful=%s, distractors=%d, issues=%s",
                valid, distinct, separated, meaningful,
                len(distractor_texts), issues,
            )

            return DistractorReport(
                valid=valid,
                distinct_distractors=distinct,
                separated_from_correct=separated,
                meaningful_distractors=meaningful,
                issues=issues,
            )
        except Exception:
            log.exception("DistractorValidation failed — blocking as safety fallback")
            return DistractorReport(
                valid=False,
                distinct_distractors=False,
                separated_from_correct=False,
                meaningful_distractors=False,
                issues=["Distractor validation raised an exception"],
            )

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.services.rag_retrieval_service import RetrievedChunk

log = get_logger(__name__)

# Words to exclude from support scoring (too short or stop words)
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "because", "but", "and", "or",
    "if", "while", "that", "this", "these", "those", "it", "its",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphabetic tokens, filtering stop words."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) >= 2]


def _content_bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    """Generate content-word bigrams from a token list."""
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def _support_score(
    text_tokens: list[str],
    evidence_tokens: set[str],
    evidence_bigrams: set[tuple[str, str]],
    *,
    min_score: float = 0.0,
) -> tuple[float, bool]:
    """Compute what fraction of *text* tokens/bigrams are covered by evidence.

    Returns ``(score, supported)`` where ``supported`` is True when
    ``score >= min_score``.  Score is 0–1.

    Empty token lists (no content words after stop-word removal) are
    treated as unsupported — caller should have provided meaningful text.
    """
    if not text_tokens:
        return 0.0, False

    covered = sum(1 for t in text_tokens if t in evidence_tokens)
    unigram_score = covered / len(text_tokens)

    bigrams = _content_bigrams(text_tokens)
    if bigrams:
        bigram_covered = sum(1 for b in bigrams if b in evidence_bigrams)
        bigram_score = bigram_covered / len(bigrams)
    else:
        bigram_score = unigram_score

    # Blend: 60% unigram, 40% bigram
    score = 0.6 * unigram_score + 0.4 * bigram_score
    return round(score, 4), score >= min_score


@dataclass
class GroundingReport:
    """Deterministic grounding validation result."""

    grounded: bool
    question_supported: bool
    answer_supported: bool
    explanation_supported: bool
    support_score: float
    issues: list[str] = field(default_factory=list)


class GroundingValidationService:
    """Validate that a generated MCQ is sufficiently grounded in evidence.

    This is a deterministic, lexical-overlap-based check — no LLM judge.
    The service examines:

    1. **Question stem support**: the question text must have meaningful
       overlap with the retrieved evidence.
    2. **Answer support**: the correct answer text must have meaningful
       overlap with the retrieved evidence.
    3. **Explanation support**: the explanation text must overlap with
       the retrieved evidence.
    4. **Overall grounding**: a blended score across all three.

    Thresholds are intentionally conservative so that borderline questions
    are flagged for human review rather than silently dropped.
    """

    # Minimum fraction of question stem that must be supported by evidence
    QUESTION_THRESHOLD: float = 0.10
    # Minimum fraction of answer text that must be supported by evidence
    ANSWER_THRESHOLD: float = 0.10
    # Minimum fraction of explanation text that must be supported
    EXPLANATION_THRESHOLD: float = 0.10
    # Minimum blended score for the question to be considered grounded
    GROUNDED_THRESHOLD: float = 0.10

    def validate(
        self,
        question_text: str,
        correct_answer_text: str,
        explanation: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> GroundingReport:
        """Run deterministic grounding checks against retrieved evidence.

        Returns a :class:`GroundingReport` with a ``grounded`` boolean
        and individual support flags.  A question is considered grounded
        only when question stem, answer, and explanation are all
        individually supported AND the overall score is above threshold.

        Exception-safe: on any unexpected error, returns an ungrounded
        report so the question is blocked (fail-closed).
        """
        issues: list[str] = []

        if not retrieved_chunks:
            return GroundingReport(
                grounded=False,
                question_supported=False,
                answer_supported=False,
                explanation_supported=False,
                support_score=0.0,
                issues=["No evidence chunks available for grounding check"],
            )

        try:
            # Combine all evidence into a single token/bigram pool
            all_evidence_tokens: set[str] = set()
            all_evidence_bigrams: set[tuple[str, str]] = set()
            for chunk in retrieved_chunks:
                tokens = _tokenize(chunk.text)
                all_evidence_tokens.update(tokens)
                all_evidence_bigrams.update(_content_bigrams(tokens))

            # --- Question stem support ---
            question_tokens = _tokenize(question_text)
            if not question_tokens:
                question_score = 0.0
                question_ok = False
                issues.append(
                    "Question stem has no content words after stop-word removal"
                )
            else:
                question_score, question_ok = _support_score(
                    question_tokens, all_evidence_tokens, all_evidence_bigrams,
                    min_score=self.QUESTION_THRESHOLD,
                )
                if not question_ok:
                    issues.append(
                        f"Question stem support below threshold "
                        f"({question_score:.2f} < {self.QUESTION_THRESHOLD})"
                    )

            # --- Answer support ---
            answer_tokens = _tokenize(correct_answer_text)
            if not answer_tokens:
                answer_score = 0.0
                answer_ok = False
                issues.append(
                    "Correct answer has no content words after stop-word removal"
                )
            else:
                answer_score, answer_ok = _support_score(
                    answer_tokens, all_evidence_tokens, all_evidence_bigrams,
                    min_score=self.ANSWER_THRESHOLD,
                )
                if not answer_ok:
                    issues.append(
                        f"Correct answer support below threshold "
                        f"({answer_score:.2f} < {self.ANSWER_THRESHOLD})"
                    )

            # --- Explanation support ---
            explanation_tokens = _tokenize(explanation)
            if not explanation_tokens:
                explanation_score = 0.0
                explanation_ok = False
                issues.append(
                    "Explanation has no content words after stop-word removal"
                )
            else:
                explanation_score, explanation_ok = _support_score(
                    explanation_tokens, all_evidence_tokens, all_evidence_bigrams,
                    min_score=self.EXPLANATION_THRESHOLD,
                )
                if not explanation_ok:
                    issues.append(
                        f"Explanation support below threshold "
                        f"({explanation_score:.2f} < {self.EXPLANATION_THRESHOLD})"
                    )

            # --- Overall grounding score ---
            overall_score = round(
                0.4 * question_score
                + 0.3 * answer_score
                + 0.3 * explanation_score,
                4,
            )
            grounded = (
                question_ok and answer_ok and explanation_ok
                and overall_score >= self.GROUNDED_THRESHOLD
            )

            log.info(
                "GroundingValidation: question=%.2f, answer=%.2f, "
                "explanation=%.2f, overall=%.2f, grounded=%s",
                question_score, answer_score, explanation_score,
                overall_score, grounded,
            )

            return GroundingReport(
                grounded=grounded,
                question_supported=question_ok,
                answer_supported=answer_ok,
                explanation_supported=explanation_ok,
                support_score=overall_score,
                issues=issues,
            )
        except Exception:
            log.exception("GroundingValidation failed — blocking as safety fallback")
            return GroundingReport(
                grounded=False,
                question_supported=False,
                answer_supported=False,
                explanation_supported=False,
                support_score=0.0,
                issues=["Grounding validation raised an exception"],
            )

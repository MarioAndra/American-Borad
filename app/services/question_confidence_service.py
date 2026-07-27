from __future__ import annotations

from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.services.generated_question_service import GenerationOutput
from app.services.generated_question_validation_service import ValidationReport
from app.services.rag_retrieval_service import RetrievedChunk

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceReport:
    """Deterministic confidence-routing decision for a generated question."""

    route: str  # "auto_approve" | "human_review" | "reject"
    score: float  # 0.0 – 100.0
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class QuestionConfidenceService:
    """Deterministic confidence-scoring before question persistence.

    Evaluates evidence quality, validation signals, retrieval context,
    and question completeness to route each generated question into one
    of three approval buckets.

    No LLM call — purely rule-based, reproducible.

    Thresholds are class constants, overridable via kwargs.
    """

    AUTO_APPROVE_THRESHOLD: float = 70.0
    HUMAN_REVIEW_THRESHOLD: float = 40.0

    def evaluate(
        self,
        *,
        retrieved_chunks: list[RetrievedChunk],
        validation_report: ValidationReport,
        retry_count: int,
        gen_output: GenerationOutput,
    ) -> ConfidenceReport:
        """Return a ConfidenceReport with route, score, and contributing reasons."""
        score = 100.0
        reasons: list[str] = []

        # --- Evidence quality (max −30) ---
        if retrieved_chunks:
            avg_sim = (
                sum(c.similarity or 0.0 for c in retrieved_chunks) / len(retrieved_chunks)
            )
            high_quality_count = sum(
                1 for c in retrieved_chunks if (c.similarity or 0.0) >= 0.5
            )

            if avg_sim < 0.3:
                score -= 15
                reasons.append(
                    f"Low avg evidence similarity ({avg_sim:.3f})"
                )
            elif avg_sim < 0.5:
                score -= 10
                reasons.append(
                    f"Moderate avg evidence similarity ({avg_sim:.3f})"
                )

            if high_quality_count == 0:
                score -= 15
                reasons.append("No high-quality evidence chunks (sim>=0.5)")
        else:
            score -= 30
            reasons.append("No retrieved chunks")

        # --- Retrieval repair penalty (−20) ---
        if retry_count > 0:
            score -= 20
            reasons.append(
                f"Retrieval repair triggered (retry_count={retry_count})"
            )

        # --- Validation near-duplicate risk (max −10) ---
        if validation_report.max_similarity > 0.5:
            score -= 10
            reasons.append(
                f"High validation similarity ({validation_report.max_similarity:.3f})"
            )
        elif validation_report.max_similarity > 0.3:
            score -= 5
            reasons.append(
                f"Moderate validation similarity ({validation_report.max_similarity:.3f})"
            )

        # --- Explanation completeness (−10) ---
        explanation = (gen_output.explanation or "").strip()
        if len(explanation) < 10:
            score -= 10
            reasons.append("Explanation too short or missing")

        # --- Question text completeness (−5) ---
        question = (gen_output.question_text or "").strip()
        if len(question) < 20:
            score -= 5
            reasons.append("Question text too short")

        # --- Clamp and route ---
        score = max(0.0, min(100.0, score))

        if score >= self.AUTO_APPROVE_THRESHOLD:
            route = "auto_approve"
        elif score >= self.HUMAN_REVIEW_THRESHOLD:
            route = "human_review"
        else:
            route = "reject"

        return ConfidenceReport(route=route, score=score, reasons=reasons)

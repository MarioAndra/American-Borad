from __future__ import annotations

from dataclasses import dataclass, field

from app.core.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_QUESTION_REPAIRS: int = 1

# Failure reasons that are eligible for a bounded repair retry.
# Terminal failures (evidence, grounding, duplicate, exceptions) abort
# immediately — repair cannot address missing evidence or hallucinated
# content with a deterministic strategy.
REPAIRABLE_FAILURES: frozenset[str] = frozenset({
    "distractor",
    "difficulty",
})


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class RepairReport:
    """Deterministic repair-decision result."""

    repairable: bool
    attempts_remaining: int
    failure_type: str
    hint: dict[str, str | float | None] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class QuestionRepairService:
    """Deterministic question-repair decision and hint generation.

    Examines the current failure state to decide:
      1. Is this failure type repairable with a deterministic strategy?
      2. Have we exhausted the repair budget?

    When repairable, produces a structured hint that the ``generate``
    node reads to adjust its behaviour on the next pass:
      - ``target_difficulty``: adjusted theta for difficulty misalignment
      - ``context_addendum``: extra instructions for distractor / general fixes

    No LLM call.  Exception-safe: on failure, returns ``repairable=False``
    so the graph aborts safely (fail-closed).

    Policy:
      - MAX_QUESTION_REPAIRS = 1 (bounded retry)
      - Repairable: distractor failure, difficulty misalignment
      - Terminal: evidence insufficiency, grounding failure, duplicate
        detection, generation exceptions, missing generation output
    """

    def decide(
        self,
        failure_reason: str | None,
        repair_attempt_count: int,
        *,
        failure_code: str | None = None,
        target_theta: float | None = None,
        difficulty_signed_delta: float | None = None,
        distractor_issues: list[str] | None = None,
    ) -> RepairReport:
        """Decide whether the current failure is repairable and within budget.

        Parameters
        ----------
        failure_code:
            Stable failure code from the graph state.  Preferred over
            ``failure_reason`` for classification when available.
        failure_reason:
            The ``failure_reason`` string from the current state, or ``None``
            if no failure is set.  Used as fallback when ``failure_code`` is
            not set.
        repair_attempt_count:
            Number of repair attempts already consumed (0 = none yet).
        target_theta:
            Student ability, used to compute adjusted difficulty target.
        difficulty_signed_delta:
            Signed delta: ``predicted_difficulty - target_theta``.
            Positive means the question is too hard; negative means too easy.
        distractor_issues:
            Issues list from DistractorReport, used for targeted hints.

        Returns
        -------
        RepairReport
            Structured result with repairability, hint, and remaining budget.
        """
        issues: list[str] = []

        try:
            # --- No failure → nothing to repair ---
            if not failure_reason and not failure_code:
                return RepairReport(
                    repairable=False,
                    attempts_remaining=max(0, MAX_QUESTION_REPAIRS - repair_attempt_count),
                    failure_type="none",
                )

            # --- Classify failure type ---
            failure_type = self._classify(failure_code=failure_code, failure_reason=failure_reason)

            # --- Terminal failure type → not repairable ---
            if failure_type not in REPAIRABLE_FAILURES:
                issues.append(f"Failure type '{failure_type}' is not repairable")
                log.info(
                    "RepairDecision BLOCKED: type=%s, code=%s, reason=%r not in REPAIRABLE_FAILURES",
                    failure_type, failure_code, failure_reason,
                )
                return RepairReport(
                    repairable=False,
                    attempts_remaining=max(0, MAX_QUESTION_REPAIRS - repair_attempt_count),
                    failure_type=failure_type,
                    issues=issues,
                )

            # --- Budget exhausted → not repairable ---
            if repair_attempt_count >= MAX_QUESTION_REPAIRS:
                issues.append(
                    f"Repair budget exhausted: {repair_attempt_count}/{MAX_QUESTION_REPAIRS}"
                )
                log.info(
                    "RepairDecision BLOCKED: budget exhausted (%d/%d)",
                    repair_attempt_count, MAX_QUESTION_REPAIRS,
                )
                return RepairReport(
                    repairable=False,
                    attempts_remaining=0,
                    failure_type=failure_type,
                    issues=issues,
                )

            # --- Repairable: build targeted hint ---
            hint = self._build_hint(
                failure_type,
                target_theta=target_theta,
                difficulty_signed_delta=difficulty_signed_delta,
                distractor_issues=distractor_issues,
            )
            remaining = MAX_QUESTION_REPAIRS - repair_attempt_count - 1

            log.info(
                "RepairDecision ALLOW: type=%s, remaining=%d, hint_keys=%s",
                failure_type, remaining, list(hint.keys()),
            )
            return RepairReport(
                repairable=True,
                attempts_remaining=remaining,
                failure_type=failure_type,
                hint=hint,
                issues=issues,
            )
        except Exception:
            log.exception("RepairDecision failed — blocking as safety fallback")
            return RepairReport(
                repairable=False,
                attempts_remaining=0,
                failure_type="unknown",
                issues=["Repair decision raised an exception"],
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Map stable failure_code → repair type.  Codes not listed here
    # are either terminal or unknown.
    _CODE_TO_TYPE: dict[str, str] = {
        "distractor_failed": "distractor",
        "difficulty_misaligned": "difficulty",
    }

    def _classify(
        self,
        *,
        failure_code: str | None = None,
        failure_reason: str | None = None,
    ) -> str:
        """Classify a failure into a coarse repair type.

        Uses ``failure_code`` (stable) when available; falls back to
        substring matching on ``failure_reason`` for backward compatibility.
        """
        if failure_code and failure_code in self._CODE_TO_TYPE:
            return self._CODE_TO_TYPE[failure_code]

        if not failure_reason:
            return "unknown"

        lower = failure_reason.lower()
        if "distractor" in lower:
            return "distractor"
        if "difficulty" in lower:
            return "difficulty"
        if "evidence" in lower:
            return "evidence"
        if "grounding" in lower:
            return "grounding"
        if "duplicate" in lower:
            return "duplicate"
        if "validation" in lower:
            return "validation"
        if "confidence" in lower:
            return "confidence"
        return "unknown"

    def _build_hint(
        self,
        failure_type: str,
        *,
        target_theta: float | None = None,
        difficulty_signed_delta: float | None = None,
        distractor_issues: list[str] | None = None,
    ) -> dict[str, str | float | None]:
        """Build a deterministic repair hint based on failure type."""
        if failure_type == "distractor":
            context = (
                "REPAIR INSTRUCTION: Your previous distractors failed quality "
                "validation. Generate 4 clearly distinct wrong answers. Each "
                "distractor must be a plausible but clearly incorrect concept "
                "related to the topic. Avoid near-duplicates and ensure every "
                "distractor is meaningfully different from the correct answer."
            )
            if distractor_issues:
                context += f" Previous issues: {'; '.join(distractor_issues)}."
            return {
                "target": "distractors",
                "context_addendum": context,
                "adjusted_theta": None,
            }

        if failure_type == "difficulty":
            adjusted_theta: float | None = None
            if target_theta is not None and difficulty_signed_delta is not None:
                # Bounded ±0.3 nudge — direction only, magnitude fixed.
                # Positive signed_delta → predicted > target → question too hard →
                #     lower theta so generator targets easier difficulty.
                # Negative signed_delta → predicted < target → question too easy →
                #     raise theta so generator targets harder difficulty.
                if difficulty_signed_delta > 0:
                    adjusted_theta = target_theta - 0.3
                elif difficulty_signed_delta < 0:
                    adjusted_theta = target_theta + 0.3
                else:
                    adjusted_theta = target_theta

            direction = "easier" if (difficulty_signed_delta or 0) > 0 else "harder"
            return {
                "target": "difficulty",
                "context_addendum": (
                    f"REPAIR INSTRUCTION: Your previous question was too "
                    f"{direction} for the target student ability. Adjust the "
                    f"difficulty accordingly."
                ),
                "adjusted_theta": adjusted_theta,
            }

        # Fallback — should not be reached for repairable types
        return {
            "target": "general",
            "context_addendum": "REPAIR INSTRUCTION: Regenerate with improved quality.",
            "adjusted_theta": None,
        }

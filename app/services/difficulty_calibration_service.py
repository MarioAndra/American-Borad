from __future__ import annotations

from dataclasses import dataclass, field

from app.core.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Difficulty bands (logit-scale IRT conventions)
# ---------------------------------------------------------------------------

EASY_UPPER: float = -0.5
HARD_LOWER: float = 0.5


def _band(value: float) -> str:
    """Map a logit-scale value to a coarse difficulty band."""
    if value < EASY_UPPER:
        return "easy"
    if value > HARD_LOWER:
        return "hard"
    return "medium"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class DifficultyCalibrationReport:
    """Result of deterministic difficulty calibration."""

    aligned: bool
    target_theta: float
    predicted_difficulty: float | None
    delta: float | None
    target_band: str
    predicted_band: str | None
    issues: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DifficultyCalibrationService:
    """Deterministic difficulty-alignment check for generated MCQs.

    Compares the target theta (student ability) against the generated
    question's ``difficulty_estimate`` and decides whether the question
    is appropriately challenging.

    Policy:
      1. If ``difficulty_estimate`` is ``None`` → block (fail-closed).
      2. Compute ``delta = abs(predicted - target)``.
      3. If ``delta > tolerance`` → block.
      4. If band mismatch and delta > band_tolerance → block.
      5. Otherwise → pass.

    All thresholds are explicit class constants.  No LLM call.
    Exception-safe: on any unexpected error, returns ``aligned=False``
    so the question is blocked (fail-closed).
    """

    # Maximum absolute delta on the logit scale before blocking
    MAX_DELTA: float = 1.5
    # Maximum delta within a band mismatch before blocking
    MAX_BAND_TOLERANCE: float = 1.0

    def calibrate(
        self,
        target_theta: float,
        difficulty_estimate: float | None,
    ) -> DifficultyCalibrationReport:
        """Run deterministic difficulty calibration.

        Parameters
        ----------
        target_theta:
            The student's current ability estimate (logit scale).
        difficulty_estimate:
            The generated question's estimated difficulty (logit scale),
            or ``None`` if unavailable.

        Returns
        -------
        DifficultyCalibrationReport
            Structured result with alignment flag, bands, delta, and issues.
        """
        issues: list[str] = []
        target_band = _band(target_theta)

        try:
            # --- Missing difficulty estimate → fail-closed ---
            if difficulty_estimate is None:
                issues.append(
                    "Missing difficulty_estimate — cannot calibrate"
                )
                log.info(
                    "DifficultyCalibration BLOCKED: missing difficulty_estimate, "
                    "target_theta=%.3f, target_band=%s",
                    target_theta, target_band,
                )
                return DifficultyCalibrationReport(
                    aligned=False,
                    target_theta=target_theta,
                    predicted_difficulty=None,
                    delta=None,
                    target_band=target_band,
                    predicted_band=None,
                    issues=issues,
                )

            predicted_band = _band(difficulty_estimate)
            delta = abs(difficulty_estimate - target_theta)

            # --- Delta exceeds absolute threshold → block ---
            if delta > self.MAX_DELTA:
                issues.append(
                    f"Delta {delta:.3f} exceeds maximum {self.MAX_DELTA}"
                )

            # --- Band mismatch with significant delta → block ---
            if target_band != predicted_band and delta > self.MAX_BAND_TOLERANCE:
                issues.append(
                    f"Band mismatch: target={target_band}, "
                    f"predicted={predicted_band} (delta={delta:.3f} > "
                    f"{self.MAX_BAND_TOLERANCE})"
                )

            aligned = len(issues) == 0

            log.info(
                "DifficultyCalibration %s: target_theta=%.3f (%s), "
                "predicted=%.3f (%s), delta=%.3f, issues=%s",
                "PASSED" if aligned else "BLOCKED",
                target_theta, target_band,
                difficulty_estimate, predicted_band,
                delta, issues,
            )

            return DifficultyCalibrationReport(
                aligned=aligned,
                target_theta=target_theta,
                predicted_difficulty=difficulty_estimate,
                delta=delta,
                target_band=target_band,
                predicted_band=predicted_band,
                issues=issues,
            )
        except Exception:
            log.exception(
                "DifficultyCalibration failed — blocking as safety fallback"
            )
            return DifficultyCalibrationReport(
                aligned=False,
                target_theta=target_theta,
                predicted_difficulty=difficulty_estimate,
                delta=None,
                target_band=target_band,
                predicted_band=None,
                issues=["Difficulty calibration raised an exception"],
            )

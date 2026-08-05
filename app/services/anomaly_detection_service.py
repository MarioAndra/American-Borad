from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.core.logging import get_logger

log = get_logger(__name__)

_BUNDLE_PATH = Path(__file__).with_name("isolation_forest_bundle.joblib")
_bundle: dict[str, Any] | None = None


def _load_bundle() -> dict[str, Any]:
    global _bundle
    if _bundle is None:
        log.info("Loading isolation forest bundle from %s", _BUNDLE_PATH)
        _bundle = joblib.load(_BUNDLE_PATH)
        log.info("Isolation forest bundle loaded (features=%s)", _bundle.get("feature_columns"))
    return _bundle


def score_response(
    *,
    student_ability: float,
    question_difficulty: float,
    elapsed_seconds: float,
) -> dict[str, Any] | None:
    """Score a single correct response for anomaly.

    Returns a dict with keys: anomaly_flag, anomaly_score, predicted_class,
    response_interpretation.  Returns ``None`` on any failure so callers
    can skip persistence without blocking the exam flow.

    Feature row order follows ``bundle["feature_columns"]`` exactly,
    matching the reference inference in ``backend_inference.py``.
    """
    try:
        bundle = _load_bundle()
        available_features = {
            "ElapsedTime": float(np.log1p(max(elapsed_seconds, 0.0))),
            "StudentAbility": float(student_ability),
            "QuestionDifficulty": float(question_difficulty),
        }
        row = np.asarray(
            [available_features[name] for name in bundle["feature_columns"]],
            dtype=float,
        )
        transformed = bundle["scaler"].transform([row])
        raw_pred = int(bundle["model"].predict(transformed)[0])
        anomaly_flag = int(raw_pred == -1)
        anomaly_score = float(-bundle["model"].decision_function(transformed)[0])
        return {
            "anomaly_flag": anomaly_flag,
            "anomaly_score": anomaly_score,
            "predicted_class": "Anomaly" if anomaly_flag else "Normal",
            "response_interpretation": (
                "Anomalous Correct Response" if anomaly_flag else "Normal Correct Response"
            ),
        }
    except Exception:
        log.exception(
            "Anomaly detection failed (ability=%.3f, difficulty=%.3f, elapsed=%.1f)",
            student_ability,
            question_difficulty,
            elapsed_seconds,
        )
        return None

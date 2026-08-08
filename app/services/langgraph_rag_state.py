from __future__ import annotations

from typing import Annotated, TypedDict

from sqlalchemy.orm import Session

from app.services.difficulty_calibration_service import DifficultyCalibrationReport
from app.services.distractor_validation_service import DistractorReport
from app.services.generated_artifact_validation_service import ArtifactValidationReport
from app.services.generated_question_service import GenerationOutput
from app.services.generated_question_validation_service import ValidationReport
from app.services.grounding_validation_service import GroundingReport
from app.services.question_confidence_service import ConfidenceReport
from app.services.question_repair_service import RepairReport
from app.services.rag_retrieval_service import RetrievedChunk
from app.services.topic_streak_service import StreakInfo


class RAGGraphState(TypedDict, total=False):
    """Typed state that flows through the LangGraph RAG workflow.

    All fields are optional (total=False) so nodes can populate them
    incrementally.  The ``db`` session is carried in state so every
    node can access it without a global dependency.
    """

    # --- Input context (set by the entrypoint) ---
    db: Session
    exam_id: int
    student_id: int
    topic_id: int
    theta: float

    # --- Gate-check output ---
    streak_info: StreakInfo | None

    # --- Retrieval outputs ---
    candidate_queries: list[str]
    retrieved_chunks: list[RetrievedChunk]

    # --- Generation outputs ---
    gen_output: GenerationOutput | None

    # --- Validation outputs ---
    validation_report: ValidationReport | None

    # --- Grounding validation output ---
    grounding_report: GroundingReport | None

    # --- Distractor validation output ---
    distractor_report: DistractorReport | None

    # --- Difficulty calibration output ---
    difficulty_report: DifficultyCalibrationReport | None

    # --- RoBERTa difficulty-model override (telemetry only) ---
    difficulty_model_report: dict | None

    # --- Artifact validation output ---
    artifact_report: ArtifactValidationReport | None

    # --- Repair metadata ---
    repair_report: RepairReport | None
    repair_attempt_count: int
    repair_hint: dict[str, str | float | None] | None

    # --- Confidence routing output ---
    confidence_report: ConfidenceReport | None

    # --- Persistence outputs ---
    generated_question_id: int | None

    # --- Flow control ---
    evidence_sufficient: bool
    failure_reason: str | None
    failure_code: str | None
    retry_count: int

    # --- Observability ---
    trace_id: str | None

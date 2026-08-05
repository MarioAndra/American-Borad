from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field

from app.models.enums import CognitiveLevel, DifficultyLevel, ExamStatus


class AdaptiveExamStartRequest(BaseModel):
    phase_id: int | None = None


class AdaptiveChoiceRead(BaseModel):
    id: int
    text: str


class AdaptiveQuestionRead(BaseModel):
    id: int
    text: str
    difficulty: DifficultyLevel
    cognitive_level: CognitiveLevel
    question_type: str
    is_generated: bool
    topic_id: int
    topic_name: str
    choices: list[AdaptiveChoiceRead]


class AdaptiveExamResult(BaseModel):
    score_percent: float
    correct_count: int
    total_questions: int
    passed: bool
    final_theta: float
    message: str


class AdaptiveExamProgressResponse(BaseModel):
    exam_id: int
    status: ExamStatus
    phase_id: int
    answered_count: int
    max_questions: int
    remaining_questions: int
    current_theta: float
    current_question: AdaptiveQuestionRead | None = None
    result: AdaptiveExamResult | None = None
    started_at: datetime
    submitted_at: datetime | None = None
    current_question_started_at: datetime | None = None


class AdaptiveAnswerRequest(BaseModel):
    question_id: int = Field(gt=0)
    choice_id: int = Field(gt=0)
    elapsed_seconds: float | None = Field(
        default=None, ge=0,
        description="Optional client-side elapsed time (non-authoritative telemetry). "
                    "The backend computes its own trusted elapsed time for anomaly detection.",
    )


class AnomalyResponseRead(BaseModel):
    response_id: int
    exam_id: int
    student_id: int
    question_id: int | None = None
    question_text: str
    is_correct: bool
    theta_before: float
    anomaly_flag: bool | None = None
    anomaly_score: float | None = None
    predicted_class: str | None = None
    response_interpretation: str | None = None
    elapsed_seconds: float | None = None
    timing_trusted: bool | None = None
    timing_issue: str | None = None
    answered_at: datetime
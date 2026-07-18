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


class AdaptiveAnswerRequest(BaseModel):
    question_id: int = Field(gt=0)
    choice_id: int = Field(gt=0)
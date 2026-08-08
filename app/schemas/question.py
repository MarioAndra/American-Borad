from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import CognitiveLevel, DifficultyLevel, QuestionType


class ChoiceCreate(BaseModel):
    text: str
    is_correct: bool = False


class ChoiceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    text: str
    is_correct: bool


class QuestionCreate(BaseModel):
    # cognitive_level is intentionally absent: it is server-owned and derived
    # from the Bloom classifier at creation time.
    text: str
    difficulty: DifficultyLevel
    question_type: QuestionType
    subtopic_id: int
    abet_criterion_id: int | None = None
    explanation: str | None = None
    common_mistake: str | None = None
    skill_gap: str | None = None
    choices: list[ChoiceCreate] = Field(min_length=2)


class QuestionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    text: str
    difficulty: DifficultyLevel
    cognitive_level: CognitiveLevel
    question_type: str
    subtopic_id: int
    abet_criterion_id: int | None
    explanation: str | None
    common_mistake: str | None
    skill_gap: str | None
    is_active: bool
    created_at: datetime
    choices: list[ChoiceRead]

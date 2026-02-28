from __future__ import annotations
from datetime import datetime
from typing import List
from pydantic import BaseModel, ConfigDict
from app.models.enums import DifficultyLevel, CognitiveLevel, ExamStatus


class ExamCreate(BaseModel):
    phase_id: int


class ChoiceRead(BaseModel):
    id: int
    text: str
    is_selected: bool


class QuestionRead(BaseModel):
    id: int
    text: str
    difficulty: DifficultyLevel
    cognitive_level: CognitiveLevel
    question_type: str
    choices: List[ChoiceRead]


class ExamRead(BaseModel):
    id: int
    phase_id: int
    total_questions: int
    easy_count: int
    medium_count: int
    hard_count: int
    status: ExamStatus
    score: float | None
    started_at: datetime | None
    submitted_at: datetime | None
    questions: List[QuestionRead]


class ExamListItem(BaseModel):
    id: int
    phase_id: int
    status: ExamStatus
    score: float | None
    created_at: datetime


class ExamSubmissionItem(BaseModel):
    question_id: int
    choice_id: int


class ExamSubmission(BaseModel):
    answers: List[ExamSubmissionItem]


class ExamResult(BaseModel):
    score_percent: float
    correct_count: int
    total_questions: int

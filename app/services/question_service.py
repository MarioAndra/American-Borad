from __future__ import annotations

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models import Choice, Question, SubTopic, User
from app.models.enums import CognitiveLevel, QuestionType
from app.schemas.question import QuestionCreate
from app.services import bloom_classifier_service

log = get_logger(__name__)


class QuestionValidationError(ValueError):
    """Invalid question payload (bad choices, missing subtopic, etc.)."""


class QuestionClassificationError(RuntimeError):
    """The Bloom classifier could not produce a cognitive level."""


class QuestionDuplicateError(ValueError):
    """A question with the same text already exists."""


# Conservative fallback used only when BLOOM_MODEL_FAIL_OPEN=true.
FALLBACK_LEVEL = CognitiveLevel.RememberUnderstand


def _validate_choices(payload: QuestionCreate) -> None:
    correct = [c for c in payload.choices if c.is_correct]
    if payload.question_type is QuestionType.SingleChoice:
        if len(correct) != 1:
            raise QuestionValidationError(
                "SingleChoice questions must have exactly one correct answer"
            )
    elif len(correct) < 1:
        raise QuestionValidationError(
            "MultipleSelect questions must have at least one correct answer"
        )
    for choice in payload.choices:
        if not choice.text.strip():
            raise QuestionValidationError("Choice text must not be empty")


def _classify(question_text: str) -> CognitiveLevel:
    settings = get_settings()
    if not settings.BLOOM_MODEL_ENABLED:
        raise QuestionClassificationError(
            "Bloom classifier is disabled (BLOOM_MODEL_ENABLED=false)"
        )
    if not settings.BLOOM_MODEL_PATH or not settings.BLOOM_MODEL_PATH.strip():
        raise QuestionClassificationError(
            "Bloom classifier is enabled but BLOOM_MODEL_PATH is not configured"
        )
    try:
        return bloom_classifier_service.predict(question_text).label
    except Exception as exc:
        if settings.BLOOM_MODEL_FAIL_OPEN:
            log.warning(
                "Bloom classification failed; persisting fallback %s: %s",
                FALLBACK_LEVEL.value,
                exc,
            )
            return FALLBACK_LEVEL
        raise QuestionClassificationError(
            "Question could not be classified"
        ) from exc


def _is_duplicate_text_error(exc: IntegrityError) -> bool:
    """True when the integrity error is the questions.text unique constraint."""
    orig = getattr(exc, "orig", None)
    diag = getattr(orig, "diag", None) if orig is not None else None
    return getattr(diag, "constraint_name", None) == "uq_questions_text"


def create_question(db: Session, payload: QuestionCreate, created_by: User) -> Question:
    if not payload.text.strip():
        raise QuestionValidationError("Question text must not be empty")
    _validate_choices(payload)

    subtopic = db.query(SubTopic).filter(SubTopic.id == payload.subtopic_id).first()
    if not subtopic:
        raise QuestionValidationError(f"SubTopic {payload.subtopic_id} not found")

    cognitive_level = _classify(payload.text)

    question = Question(
        text=payload.text.strip(),
        difficulty=payload.difficulty,
        cognitive_level=cognitive_level,
        question_type=payload.question_type.value,
        subtopic_id=payload.subtopic_id,
        abet_criterion_id=payload.abet_criterion_id,
        created_by=created_by.id,
        is_active=True,
        explanation=payload.explanation,
        common_mistake=payload.common_mistake,
        skill_gap=payload.skill_gap,
    )
    try:
        db.add(question)
        db.flush()

        for choice in payload.choices:
            db.add(
                Choice(
                    question_id=question.id,
                    text=choice.text.strip(),
                    is_correct=choice.is_correct,
                )
            )

        db.commit()
    except IntegrityError as exc:
        db.rollback()
        if _is_duplicate_text_error(exc):
            raise QuestionDuplicateError(
                "A question with this text already exists"
            ) from None
        raise QuestionValidationError(
            "Question data violates database constraints"
        ) from exc
    db.refresh(question)
    return question


def get_question_by_id(db: Session, question_id: int) -> Question | None:
    return db.query(Question).filter(Question.id == question_id).first()

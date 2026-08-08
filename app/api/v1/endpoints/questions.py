from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.dependencies import require_roles
from app.db.session import get_db
from app.models import User
from app.schemas.question import QuestionCreate, QuestionResponse
from app.services import question_service

router = APIRouter()


@router.post(
    "",
    response_model=QuestionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_question(
    payload: QuestionCreate,
    admin: User = Depends(require_roles("Admin")),
    db: Session = Depends(get_db),
) -> QuestionResponse:
    try:
        question = question_service.create_question(db, payload, created_by=admin)
    except question_service.QuestionValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except question_service.QuestionDuplicateError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except question_service.QuestionClassificationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return question

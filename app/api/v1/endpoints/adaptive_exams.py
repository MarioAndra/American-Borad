from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, require_roles
from app.db.session import get_db
from app.models import User
from app.schemas.adaptive_exam import (
    AdaptiveAnswerRequest,
    AdaptiveExamProgressResponse,
    AdaptiveExamStartRequest,
)
from app.services.adaptive_exam_service import get_adaptive_exam, start_adaptive_exam, submit_adaptive_answer

router = APIRouter()


@router.post(
    "/exams/start",
    response_model=AdaptiveExamProgressResponse,
    dependencies=[Depends(require_roles("Student"))],
)
def start_phase2_exam(
    payload: AdaptiveExamStartRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AdaptiveExamProgressResponse:
    try:
        data = start_adaptive_exam(db, user.id, payload.phase_id)
        return AdaptiveExamProgressResponse(**data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post(
    "/exams/{exam_id}/answer",
    response_model=AdaptiveExamProgressResponse,
    dependencies=[Depends(require_roles("Student"))],
)
def answer_phase2_exam(
    exam_id: int,
    payload: AdaptiveAnswerRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AdaptiveExamProgressResponse:
    try:
        data = submit_adaptive_answer(db, user.id, exam_id, payload.question_id, payload.choice_id)
        return AdaptiveExamProgressResponse(**data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get(
    "/exams/{exam_id}",
    response_model=AdaptiveExamProgressResponse,
    dependencies=[Depends(require_roles("Student"))],
)
def get_phase2_exam(
    exam_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AdaptiveExamProgressResponse:
    try:
        data = get_adaptive_exam(db, user.id, exam_id)
        return AdaptiveExamProgressResponse(**data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
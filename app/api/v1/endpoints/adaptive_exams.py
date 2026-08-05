from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, require_roles
from app.db.session import get_db
from app.models import AdaptiveExamResponse, User
from app.schemas.adaptive_exam import (
    AdaptiveAnswerRequest,
    AdaptiveExamProgressResponse,
    AdaptiveExamStartRequest,
    AnomalyResponseRead,
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
        data = submit_adaptive_answer(
            db, user.id, exam_id, payload.question_id, payload.choice_id,
        )
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


# ── Admin: anomaly review endpoints ────────────────────────────────


@router.get(
    "/admin/anomalies",
    response_model=list[AnomalyResponseRead],
    dependencies=[Depends(require_roles("Admin"))],
)
def list_anomalies(
    flagged_only: bool = Query(default=True, description="If true, only return anomaly-flagged responses"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[AnomalyResponseRead]:
    q = (
        db.query(AdaptiveExamResponse)
        .filter(AdaptiveExamResponse.anomaly_flag.isnot(None))
    )
    if flagged_only:
        q = q.filter(AdaptiveExamResponse.anomaly_flag == True)  # noqa: E712
    rows = (
        q.order_by(AdaptiveExamResponse.answered_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [_to_anomaly_read(r) for r in rows]


@router.get(
    "/admin/exams/{exam_id}/anomalies",
    response_model=list[AnomalyResponseRead],
    dependencies=[Depends(require_roles("Admin"))],
)
def list_exam_anomalies(
    exam_id: int,
    db: Session = Depends(get_db),
) -> list[AnomalyResponseRead]:
    rows = (
        db.query(AdaptiveExamResponse)
        .filter(
            AdaptiveExamResponse.adaptive_exam_id == exam_id,
            AdaptiveExamResponse.anomaly_flag.isnot(None),
        )
        .order_by(AdaptiveExamResponse.order_index.asc())
        .all()
    )
    return [_to_anomaly_read(r) for r in rows]


def _to_anomaly_read(resp: AdaptiveExamResponse) -> AnomalyResponseRead:
    question_text = ""
    if resp.question_id is not None:
        q = resp.question  # already loaded via relationship
        question_text = q.text if q else ""
    return AnomalyResponseRead(
        response_id=resp.id,
        exam_id=resp.adaptive_exam_id,
        student_id=resp.exam.student_id,
        question_id=resp.question_id,
        question_text=question_text,
        is_correct=resp.is_correct,
        theta_before=resp.theta_before,
        anomaly_flag=resp.anomaly_flag,
        anomaly_score=resp.anomaly_score,
        predicted_class=resp.predicted_class,
        response_interpretation=resp.response_interpretation,
        elapsed_seconds=resp.elapsed_seconds,
        timing_trusted=resp.timing_trusted,
        timing_issue=resp.timing_issue,
        answered_at=resp.answered_at,
    )
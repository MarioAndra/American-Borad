from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, require_roles
from app.core.logging import get_logger
from app.db.session import get_db
from app.models import GeneratedQuestion, User
from app.models.enums import GeneratedQuestionStatus
from app.schemas.rag import (
    GeneratedQuestionListResponse,
    GeneratedQuestionRead,
    IngestRequest,
    IngestResponse,
    ReviewRequest,
    VerifyResponse,
)
from app.services.rag_ingestion_service import RAGIngestionService

log = get_logger(__name__)
router = APIRouter()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(require_roles("Admin"))],
)
def trigger_ingestion(
    payload: IngestRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> IngestResponse:
    service = RAGIngestionService(db)
    try:
        if payload.course_name:
            result = service.ingest_course(payload.course_name, force=payload.force)
        else:
            result = service.ingest_all(force=payload.force)
        return IngestResponse(
            documents_created=result.documents_created,
            chunks_created=result.chunks_created,
            chunks_embedded=result.chunks_embedded,
            errors=result.errors,
            quarantined=result.quarantined,
            zero_chunk_docs=result.zero_chunk_docs,
        )
    finally:
        service.close()


@router.post(
    "/extract",
    response_model=IngestResponse,
    dependencies=[Depends(require_roles("Admin"))],
)
def trigger_extraction(
    payload: IngestRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> IngestResponse:
    """Phase 1: extract text from PDFs and chunk — no API calls."""
    service = RAGIngestionService(db)
    try:
        if payload.course_name:
            result = service.extract_and_chunk_course(payload.course_name, force=payload.force)
        else:
            result = service.extract_and_chunk_all(force=payload.force)
        return IngestResponse(
            documents_created=result.documents_created,
            chunks_created=result.chunks_created,
            chunks_embedded=result.chunks_embedded,
            errors=result.errors,
            quarantined=result.quarantined,
            zero_chunk_docs=result.zero_chunk_docs,
        )
    finally:
        service.close()


@router.post(
    "/embed",
    response_model=IngestResponse,
    dependencies=[Depends(require_roles("Admin"))],
)
def trigger_embedding(
    payload: IngestRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> IngestResponse:
    """Phase 2: embed pending/failed chunks and store in Weaviate — makes API calls."""
    service = RAGIngestionService(db)
    try:
        if payload.course_name:
            result = service.embed_pending_for_course(payload.course_name)
        else:
            result = service.embed_all_pending()
        return IngestResponse(
            documents_created=result.documents_created,
            chunks_created=result.chunks_created,
            chunks_embedded=result.chunks_embedded,
            errors=result.errors,
            quarantined=result.quarantined,
            zero_chunk_docs=result.zero_chunk_docs,
        )
    finally:
        service.close()


@router.post(
    "/check",
    dependencies=[Depends(require_roles("Admin"))],
)
def check_readiness(
    smoke_test: bool = False,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Validate all pre-conditions for ingestion. Set smoke_test=true for a live embed+search."""
    service = RAGIngestionService(db)
    try:
        return service.readiness_check(smoke_test=smoke_test)
    finally:
        service.close()


@router.post(
    "/verify",
    response_model=VerifyResponse,
    dependencies=[Depends(require_roles("Admin"))],
)
def run_verification(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Compare PG docs/chunks vs Weaviate and run sample retrievals per topic."""
    service = RAGIngestionService(db)
    try:
        return service.verify()
    finally:
        service.close()


@router.get(
    "/questions",
    response_model=GeneratedQuestionListResponse,
    dependencies=[Depends(require_roles("Admin"))],
)
def list_generated_questions(
    topic_id: int | None = Query(None),
    status: str | None = Query(None, pattern="^(draft|approved|rejected|auto_approved)$"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> GeneratedQuestionListResponse:
    q = db.query(GeneratedQuestion)
    if topic_id is not None:
        q = q.filter(GeneratedQuestion.topic_id == topic_id)
    if status is not None:
        q = q.filter(GeneratedQuestion.status == status)

    total = q.count()
    items = q.order_by(GeneratedQuestion.created_at.desc()).offset((page - 1) * per_page).limit(per_page).all()

    return GeneratedQuestionListResponse(
        items=[GeneratedQuestionRead(**{
            "id": gq.id,
            "topic_id": gq.topic_id,
            "text": gq.text,
            "choices": gq.choices,
            "explanation": gq.explanation,
            "difficulty_estimate": gq.difficulty_estimate,
            "status": gq.status,
            "review_required": gq.review_required,
            "validation_report": gq.validation_report,
            "created_at": gq.created_at,
            "is_generated": True,
        }) for gq in items],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.patch(
    "/questions/{question_id}/review",
    response_model=GeneratedQuestionRead,
    dependencies=[Depends(require_roles("Admin"))],
)
def review_generated_question(
    question_id: int,
    payload: ReviewRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> GeneratedQuestionRead:
    gq = db.query(GeneratedQuestion).filter(GeneratedQuestion.id == question_id).first()
    if not gq:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generated question not found")

    from app.utils.common import utcnow
    from app.models.rag import GeneratedQuestionReview

    review = GeneratedQuestionReview(
        generated_question_id=gq.id,
        reviewer_id=user.id,
        decision=payload.decision,
        comments=payload.comments,
    )
    db.add(review)

    new_status = GeneratedQuestionStatus(payload.decision)
    gq.status = new_status
    gq.review_required = False
    db.add(gq)
    db.flush()
    db.commit()
    db.refresh(gq)

    return GeneratedQuestionRead(**{
        "id": gq.id,
        "topic_id": gq.topic_id,
        "text": gq.text,
        "choices": gq.choices,
        "explanation": gq.explanation,
        "difficulty_estimate": gq.difficulty_estimate,
        "status": gq.status,
        "review_required": gq.review_required,
        "validation_report": gq.validation_report,
        "created_at": gq.created_at,
        "is_generated": True,
    })

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.models.enums import GeneratedQuestionStatus


class IngestRequest(BaseModel):
    course_name: str | None = None
    force: bool = False


class VerifyResponse(BaseModel):
    pg_documents: int = 0
    pg_chunks: int = 0
    weaviate_objects: int = 0
    objects_orphaned: int = 0
    chunks_missing: list[int] = []
    topic_coverage: dict[str, int] = {}
    sample_searches: list[dict] = []
    errors: list[str] = []


class IngestResponse(BaseModel):
    documents_created: int = 0
    chunks_created: int = 0
    chunks_embedded: int = 0
    errors: list[str] = []
    quarantined: list[str] = []
    zero_chunk_docs: list[int] = []


class GeneratedQuestionRead(BaseModel):
    id: int
    topic_id: int
    text: str
    choices: list[dict]
    explanation: str
    difficulty_estimate: float | None
    status: GeneratedQuestionStatus
    review_required: bool
    validation_report: dict | None
    created_at: datetime
    is_generated: bool = True


class GeneratedQuestionListResponse(BaseModel):
    items: list[GeneratedQuestionRead]
    total: int
    page: int
    per_page: int


class ReviewRequest(BaseModel):
    decision: str = Field(pattern="^(approved|rejected)$")
    comments: str | None = None

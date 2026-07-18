from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum as SAEnum, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import GeneratedQuestionStatus


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    course_name: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    topic_id: Mapped[int | None] = mapped_column(ForeignKey("topics.id", ondelete="SET NULL"), nullable=True, index=True)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False)
    embedding_status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", server_default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    chunks: Mapped[list["KnowledgeChunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    topic_id: Mapped[int | None] = mapped_column(ForeignKey("topics.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    document: Mapped["KnowledgeDocument"] = relationship(back_populates="chunks")


class GeneratedQuestion(Base):
    __tablename__ = "generated_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id", ondelete="RESTRICT"), nullable=False, index=True)
    source_exam_id: Mapped[int | None] = mapped_column(ForeignKey("adaptive_exams.id", ondelete="SET NULL"), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    choices: Mapped[dict] = mapped_column(JSON, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    difficulty_estimate: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[GeneratedQuestionStatus] = mapped_column(
        SAEnum(GeneratedQuestionStatus, name="generated_question_status"),
        nullable=False,
        index=True,
    )
    review_required: Mapped[bool] = mapped_column(nullable=False, default=True)
    validation_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    evidence: Mapped[list["GeneratedQuestionEvidence"]] = relationship(back_populates="question", cascade="all, delete-orphan")
    reviews: Mapped[list["GeneratedQuestionReview"]] = relationship(back_populates="question", cascade="all, delete-orphan")


class GeneratedQuestionEvidence(Base):
    __tablename__ = "generated_question_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    generated_question_id: Mapped[int] = mapped_column(
        ForeignKey("generated_questions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_chunks.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    question: Mapped["GeneratedQuestion"] = relationship(back_populates="evidence")


class GeneratedQuestionReview(Base):
    __tablename__ = "generated_question_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    generated_question_id: Mapped[int] = mapped_column(
        ForeignKey("generated_questions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    reviewer_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    comments: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    question: Mapped["GeneratedQuestion"] = relationship(back_populates="reviews")


class StudentTopicProgress(Base):
    __tablename__ = "student_topic_progress"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("adaptive_exams.id", ondelete="CASCADE"), nullable=False, index=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id", ondelete="RESTRICT"), nullable=False, index=True)
    current_streak: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    questions_asked: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    generated_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_theta: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_theta: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    consumed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

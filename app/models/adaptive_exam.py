from __future__ import annotations

from sqlalchemy import CheckConstraint, DateTime, Enum as SAEnum, Float, ForeignKey, Integer, Numeric, func, text as sa_text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import ExamStatus


class AdaptiveExam(Base):
    __tablename__ = "adaptive_exams"
    __table_args__ = (
        CheckConstraint("answered_count <= max_questions", name="ck_adaptive_exam_answered_max"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    phase_id: Mapped[int] = mapped_column(ForeignKey("phases.id", ondelete="RESTRICT"), nullable=False, index=True)
    status: Mapped[ExamStatus] = mapped_column(SAEnum(ExamStatus, name="exam_status"), nullable=False, index=True)
    max_questions: Mapped[int] = mapped_column(Integer, nullable=False)
    answered_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=sa_text("0"))
    current_theta: Mapped[float] = mapped_column(Float, nullable=False, server_default=sa_text("0"))
    score: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    started_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    submitted_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    locked_topic_id: Mapped[int | None] = mapped_column(
        ForeignKey("topics.id", ondelete="SET NULL"), nullable=True, index=True,
    )
    pending_generated_question_id: Mapped[int | None] = mapped_column(
        ForeignKey("generated_questions.id", ondelete="SET NULL"), nullable=True, index=True,
    )
    current_question_started_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    current_question_id: Mapped[int | None] = mapped_column(
        ForeignKey("questions.id", ondelete="SET NULL"), nullable=True, index=True,
    )
    current_generated_question_id: Mapped[int | None] = mapped_column(
        ForeignKey("generated_questions.id", ondelete="SET NULL"), nullable=True, index=True,
    )

    responses: Mapped[list["AdaptiveExamResponse"]] = relationship(
        back_populates="exam",
        cascade="all, delete-orphan",
        order_by="AdaptiveExamResponse.order_index",
    )
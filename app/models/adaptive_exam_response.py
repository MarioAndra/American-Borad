from __future__ import annotations

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, UniqueConstraint, func, text as sa_text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class AdaptiveExamResponse(Base):
    __tablename__ = "adaptive_exam_responses"
    __table_args__ = (
        UniqueConstraint("adaptive_exam_id", "question_id", name="uq_adaptive_exam_question"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    adaptive_exam_id: Mapped[int] = mapped_column(ForeignKey("adaptive_exams.id", ondelete="CASCADE"), nullable=False, index=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id", ondelete="RESTRICT"), nullable=False, index=True)
    choice_id: Mapped[int] = mapped_column(ForeignKey("choices.id", ondelete="RESTRICT"), nullable=False, index=True)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=sa_text("false"))
    theta_before: Mapped[float] = mapped_column(Float, nullable=False)
    theta_after: Mapped[float] = mapped_column(Float, nullable=False)
    answered_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    exam: Mapped["AdaptiveExam"] = relationship(back_populates="responses")
    question: Mapped["Question"] = relationship()
    choice: Mapped["Choice"] = relationship()
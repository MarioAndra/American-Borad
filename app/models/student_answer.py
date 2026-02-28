from __future__ import annotations

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, UniqueConstraint, func, text as sa_text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class StudentAnswer(Base):
    __tablename__ = "student_answers"
    __table_args__ = (UniqueConstraint("exam_id", "question_id", "choice_id", name="uq_student_answers_sel"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id", ondelete="CASCADE"), nullable=False, index=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id", ondelete="RESTRICT"), nullable=False, index=True)
    choice_id: Mapped[int] = mapped_column(ForeignKey("choices.id", ondelete="RESTRICT"), nullable=False, index=True)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=sa_text("false"))
    answered_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    exam: Mapped["Exam"] = relationship(back_populates="student_answers")
    question: Mapped["Question"] = relationship()
    choice: Mapped["Choice"] = relationship(back_populates="student_answers")

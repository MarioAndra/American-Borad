from __future__ import annotations

from sqlalchemy import ForeignKey, Integer, PrimaryKeyConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class ExamQuestion(Base):
    __tablename__ = "exam_questions"
    __table_args__ = (PrimaryKeyConstraint("exam_id", "question_id", name="pk_exam_questions"),)

    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id", ondelete="CASCADE"), nullable=False, index=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id", ondelete="RESTRICT"), nullable=False, index=True)
    position: Mapped[int | None] = mapped_column(Integer, nullable=True)

    exam: Mapped["Exam"] = relationship(back_populates="exam_questions")
    question: Mapped["Question"] = relationship()

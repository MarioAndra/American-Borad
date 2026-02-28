from __future__ import annotations

from sqlalchemy import CheckConstraint, DateTime, Enum as SAEnum, ForeignKey, Integer, Numeric, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import ExamStatus


class Exam(Base):
    __tablename__ = "exams"
    __table_args__ = (
        CheckConstraint("easy_count + medium_count + hard_count = total_questions", name="ck_exam_counts_total"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    phase_id: Mapped[int] = mapped_column(ForeignKey("phases.id", ondelete="RESTRICT"), nullable=False, index=True)
    total_questions: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    easy_count: Mapped[int] = mapped_column(Integer, nullable=False)
    medium_count: Mapped[int] = mapped_column(Integer, nullable=False)
    hard_count: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[ExamStatus] = mapped_column(SAEnum(ExamStatus, name="exam_status"), nullable=False, index=True)
    score: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    started_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    submitted_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    student: Mapped["User"] = relationship(back_populates="exams")
    phase: Mapped["Phase"] = relationship(back_populates="exams")
    exam_questions: Mapped[list["ExamQuestion"]] = relationship(back_populates="exam", cascade="all, delete-orphan")
    questions: Mapped[list["Question"]] = relationship(secondary="exam_questions", back_populates="exams")
    student_answers: Mapped[list["StudentAnswer"]] = relationship(back_populates="exam", cascade="all, delete-orphan")

from __future__ import annotations

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text, UniqueConstraint, text as sa_text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Choice(Base):
    __tablename__ = "choices"
    __table_args__ = (UniqueConstraint("question_id", "text", name="uq_choices_question_text"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id", ondelete="CASCADE"), nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=sa_text("false"))

    question: Mapped["Question"] = relationship(back_populates="choices")
    student_answers: Mapped[list["StudentAnswer"]] = relationship(back_populates="choice")

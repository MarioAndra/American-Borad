from __future__ import annotations

from sqlalchemy import Boolean, DateTime, Enum as SAEnum, Float, ForeignKey, Integer, String, Text, func, text as sa_text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import CognitiveLevel, DifficultyLevel


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    difficulty: Mapped[DifficultyLevel] = mapped_column(
        SAEnum(DifficultyLevel, name="difficulty_level"),
        nullable=False,
        index=True,
    )
    cognitive_level: Mapped[CognitiveLevel] = mapped_column(
        SAEnum(CognitiveLevel, name="cognitive_level"),
        nullable=False,
        index=True,
    )
    question_type: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    irt_a: Mapped[float | None] = mapped_column(Float, nullable=True)
    irt_b: Mapped[float | None] = mapped_column(Float, nullable=True)
    irt_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    subtopic_id: Mapped[int] = mapped_column(
        ForeignKey("subtopics.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    abet_criterion_id: Mapped[int | None] = mapped_column(
        ForeignKey("abet_criteria.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    created_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=sa_text("true"),
    )
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    common_mistake: Mapped[str | None] = mapped_column(Text, nullable=True)
    skill_gap: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    choices: Mapped[list["Choice"]] = relationship(
        back_populates="question",
        cascade="all, delete-orphan",
    )
    subtopic: Mapped["SubTopic"] = relationship(back_populates="questions")
    abet_criterion: Mapped["ABETCriterion"] = relationship(back_populates="questions")
    created_by_user: Mapped["User"] = relationship(back_populates="questions_created")
    exams: Mapped[list["Exam"]] = relationship(
        secondary="exam_questions",
        back_populates="questions",
    )

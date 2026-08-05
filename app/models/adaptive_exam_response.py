from __future__ import annotations

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, func, text as sa_text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class AdaptiveExamResponse(Base):
    __tablename__ = "adaptive_exam_responses"
    __table_args__ = (
        UniqueConstraint("adaptive_exam_id", "question_id", name="uq_adaptive_exam_question"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    adaptive_exam_id: Mapped[int] = mapped_column(ForeignKey("adaptive_exams.id", ondelete="CASCADE"), nullable=False, index=True)
    question_id: Mapped[int | None] = mapped_column(ForeignKey("questions.id", ondelete="RESTRICT"), nullable=True, index=True)
    generated_question_id: Mapped[int | None] = mapped_column(ForeignKey("generated_questions.id", ondelete="SET NULL"), nullable=True, index=True)
    choice_id: Mapped[int | None] = mapped_column(ForeignKey("choices.id", ondelete="RESTRICT"), nullable=True, index=True)
    selected_option_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=sa_text("false"))
    theta_before: Mapped[float] = mapped_column(Float, nullable=False)
    theta_after: Mapped[float] = mapped_column(Float, nullable=False)
    answered_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # ── Anomaly detection fields (Phase II only) ────────────────────
    anomaly_flag: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_class: Mapped[str | None] = mapped_column(String(16), nullable=True)
    response_interpretation: Mapped[str | None] = mapped_column(String(64), nullable=True)
    elapsed_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Timing trust state (Phase II only) ──────────────────────────
    # timing_trusted=False means elapsed_seconds was NOT measured from a
    # server-owned serve timestamp attributable to this exact question,
    # so anomaly scoring was skipped.  timing_issue records the reason.
    timing_trusted: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    timing_issue: Mapped[str | None] = mapped_column(String(64), nullable=True)

    exam: Mapped["AdaptiveExam"] = relationship(back_populates="responses")
    question: Mapped["Question"] = relationship()
    choice: Mapped["Choice"] = relationship()
    generated_question: Mapped["GeneratedQuestion | None"] = relationship()
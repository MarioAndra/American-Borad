from __future__ import annotations

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Topic(Base):
    __tablename__ = "topics"
    __table_args__ = (UniqueConstraint("phase_id", "name", name="uq_topics_phase_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    phase_id: Mapped[int] = mapped_column(ForeignKey("phases.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    phase: Mapped["Phase"] = relationship(back_populates="topics")
    subtopics: Mapped[list["SubTopic"]] = relationship(back_populates="topic", cascade="all, delete-orphan")

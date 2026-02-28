from __future__ import annotations

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class SubTopic(Base):
    __tablename__ = "subtopics"
    __table_args__ = (UniqueConstraint("topic_id", "name", name="uq_subtopics_topic_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    topic: Mapped["Topic"] = relationship(back_populates="subtopics")
    questions: Mapped[list["Question"]] = relationship(back_populates="subtopic")

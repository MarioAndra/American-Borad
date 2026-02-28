from __future__ import annotations

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Phase(Base):
    __tablename__ = "phases"
    __table_args__ = (UniqueConstraint("name", name="uq_phases_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    topics: Mapped[list["Topic"]] = relationship(back_populates="phase", cascade="all, delete-orphan")
    exams: Mapped[list["Exam"]] = relationship(back_populates="phase")


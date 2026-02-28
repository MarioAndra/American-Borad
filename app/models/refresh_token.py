from __future__ import annotations

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    __table_args__ = (UniqueConstraint("jti", name="uq_refresh_token_jti"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    jti: Mapped[str] = mapped_column(String(255), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    parent_jti: Mapped[str | None] = mapped_column(String(255), nullable=True)
    replaced_by_jti: Mapped[str | None] = mapped_column(String(255), nullable=True)
    revoked: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    expires_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

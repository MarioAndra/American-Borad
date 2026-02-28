from __future__ import annotations

from sqlalchemy import Boolean, DateTime, Enum as SAEnum, Integer, String, func, text as sa_text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import UserRole


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(SAEnum(UserRole, name="user_role"), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=sa_text("true"))
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=sa_text("false"))
    verification_token: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    verification_sent_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    verified_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    verification_code_hash: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    verification_code_expires_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    verification_code_attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default=sa_text("0"))

    failed_login_attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default=sa_text("0"))
    last_failed_login_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    locked_until: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    password_changed_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    password_reset_token_hash: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    password_reset_sent_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    password_reset_expires_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    exams: Mapped[list["Exam"]] = relationship(back_populates="student")
    questions_created: Mapped[list["Question"]] = relationship(back_populates="created_by_user")
    token_blacklist_entries: Mapped[list["TokenBlacklist"]] = relationship(back_populates="user")

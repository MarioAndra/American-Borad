"""add user security fields
Revision ID: 003
Revises: 002
Create Date: 2026-02-23
"""
from alembic import op
import sqlalchemy as sa

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("users", sa.Column("failed_login_attempts", sa.Integer(), nullable=False, server_default=sa.text("0")))
    op.add_column("users", sa.Column("last_failed_login_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("password_changed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("password_reset_token_hash", sa.String(128), nullable=True))
    op.add_column("users", sa.Column("password_reset_sent_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("password_reset_expires_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_users_password_reset_token_hash", "users", ["password_reset_token_hash"], unique=False)


def downgrade():
    op.drop_index("ix_users_password_reset_token_hash", table_name="users")
    op.drop_column("users", "password_reset_expires_at")
    op.drop_column("users", "password_reset_sent_at")
    op.drop_column("users", "password_reset_token_hash")
    op.drop_column("users", "password_changed_at")
    op.drop_column("users", "locked_until")
    op.drop_column("users", "last_failed_login_at")
    op.drop_column("users", "failed_login_attempts")

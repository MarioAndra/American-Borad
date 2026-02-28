"""add email OTP fields to users
Revision ID: 002
Revises: 001
Create Date: 2026-02-23
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("users", sa.Column("verification_code_hash", sa.String(128), nullable=True))
    op.add_column("users", sa.Column("verification_code_expires_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "users",
        sa.Column("verification_code_attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.create_index("ix_users_verification_code_hash", "users", ["verification_code_hash"], unique=False)


def downgrade():
    op.drop_index("ix_users_verification_code_hash", table_name="users")
    op.drop_column("users", "verification_code_attempts")
    op.drop_column("users", "verification_code_expires_at")
    op.drop_column("users", "verification_code_hash")

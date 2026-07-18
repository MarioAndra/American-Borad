"""add generated_question_id and selected_option_index to adaptive_exam_responses

Revision ID: 007
Revises: 006
Create Date: 2026-05-20
"""
from alembic import op
import sqlalchemy as sa

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "adaptive_exam_responses",
        sa.Column("generated_question_id", sa.Integer(), sa.ForeignKey("generated_questions.id", ondelete="SET NULL"), nullable=True),
    )
    op.add_column(
        "adaptive_exam_responses",
        sa.Column("selected_option_index", sa.Integer(), nullable=True),
    )
    op.create_index("ix_aer_generated_question_id", "adaptive_exam_responses", ["generated_question_id"], unique=False)
    op.alter_column("adaptive_exam_responses", "question_id", nullable=True)
    op.alter_column("adaptive_exam_responses", "choice_id", nullable=True)


def downgrade() -> None:
    op.drop_index("ix_aer_generated_question_id", table_name="adaptive_exam_responses")
    op.drop_column("adaptive_exam_responses", "selected_option_index")
    op.drop_column("adaptive_exam_responses", "generated_question_id")
    op.alter_column("adaptive_exam_responses", "question_id", nullable=False)
    op.alter_column("adaptive_exam_responses", "choice_id", nullable=False)

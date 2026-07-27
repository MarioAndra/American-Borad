"""add pending_generated_question_id to adaptive_exams

Revision ID: 011
Revises: 010
Create Date: 2026-07-26 13:00:00.000000

Adds ``adaptive_exams.pending_generated_question_id`` — nullable FK to
``generated_questions.id``.  When a generated question has been served as
the next question but the student has not yet answered it, this column
stores the reference so that ``GET /phase2/exams/{id}`` can return the
same pending question after a page refresh instead of recomputing a
regular next question.

Cleared when:
- the student answers the generated question (topic consumed, lock cleared)
- the exam completes
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '011'
down_revision: Union[str, None] = '010'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "adaptive_exams",
        sa.Column(
            "pending_generated_question_id",
            sa.Integer,
            sa.ForeignKey("generated_questions.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_adaptive_exams_pending_generated_question_id",
        "adaptive_exams",
        ["pending_generated_question_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_adaptive_exams_pending_generated_question_id",
        table_name="adaptive_exams",
    )
    op.drop_column("adaptive_exams", "pending_generated_question_id")

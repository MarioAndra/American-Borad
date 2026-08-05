"""add current question timing identity and trust fields

Revision ID: 014
Revises: 013
Create Date: 2026-08-05 13:00:00.000000

Makes the Phase II serve timer attributable to an exact question and
records whether each response's elapsed time is trustworthy.

``adaptive_exams`` gains:
- ``current_question_id`` (FK questions.id) — the exact regular question
  the ``current_question_started_at`` serve timer belongs to.
- ``current_generated_question_id`` (FK generated_questions.id) — the
  exact generated question the serve timer belongs to.

``adaptive_exam_responses`` gains:
- ``timing_trusted`` (bool) — True when elapsed_seconds was measured from
  a server-owned serve timestamp attributable to the answered question.
- ``timing_issue`` (str) — audit reason when timing is untrusted
  ("no_tracked_question", "question_mismatch", "missing_serve_timestamp").

All new columns are nullable so existing rows are unaffected and no data
is destroyed.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '014'
down_revision: Union[str, None] = '013'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "adaptive_exams",
        sa.Column(
            "current_question_id",
            sa.Integer,
            sa.ForeignKey("questions.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_adaptive_exams_current_question_id",
        "adaptive_exams",
        ["current_question_id"],
    )
    op.add_column(
        "adaptive_exams",
        sa.Column(
            "current_generated_question_id",
            sa.Integer,
            sa.ForeignKey("generated_questions.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_adaptive_exams_current_generated_question_id",
        "adaptive_exams",
        ["current_generated_question_id"],
    )
    op.add_column(
        "adaptive_exam_responses",
        sa.Column("timing_trusted", sa.Boolean, nullable=True),
    )
    op.add_column(
        "adaptive_exam_responses",
        sa.Column("timing_issue", sa.String(64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("adaptive_exam_responses", "timing_issue")
    op.drop_column("adaptive_exam_responses", "timing_trusted")
    op.drop_index(
        "ix_adaptive_exams_current_generated_question_id",
        table_name="adaptive_exams",
    )
    op.drop_column("adaptive_exams", "current_generated_question_id")
    op.drop_index(
        "ix_adaptive_exams_current_question_id",
        table_name="adaptive_exams",
    )
    op.drop_column("adaptive_exams", "current_question_id")

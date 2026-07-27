"""add topic-lock and generation-attempted fields

Revision ID: 010
Revises: 009
Create Date: 2026-07-26 12:00:00.000000

Adds two columns needed for the explicit same-topic lock/policy
in Phase II orchestration:

1. ``adaptive_exams.locked_topic_id`` — nullable FK to ``topics.id``.
   When set, the next regular question must come from this topic.
   Cleared when the topic is consumed, runs out of eligible questions,
   or the student switches topics.

2. ``student_topic_progress.generation_attempted`` — boolean (default
   False).  Set to True after the first generation attempt (success or
   failure) for the current streak, preventing infinite retry loops.
   Reset to False when the streak resets (topic change).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '010'
down_revision: Union[str, None] = '009'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "adaptive_exams",
        sa.Column(
            "locked_topic_id",
            sa.Integer,
            sa.ForeignKey("topics.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_adaptive_exams_locked_topic_id",
        "adaptive_exams",
        ["locked_topic_id"],
    )

    op.add_column(
        "student_topic_progress",
        sa.Column(
            "generation_attempted",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("false"),
        ),
    )


def downgrade() -> None:
    op.drop_column("student_topic_progress", "generation_attempted")
    op.drop_index("ix_adaptive_exams_locked_topic_id", table_name="adaptive_exams")
    op.drop_column("adaptive_exams", "locked_topic_id")

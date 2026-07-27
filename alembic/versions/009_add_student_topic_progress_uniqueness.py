"""add unique constraint to student_topic_progress

Revision ID: 009
Revises: 008
Create Date: 2026-07-26 00:00:00.000000

Duplicate cleanup strategy:
    Before adding the unique constraint, we must reconcile any existing
    duplicate (student_id, exam_id, topic_id) rows.  The strategy is
    deterministic:

    1. Identify all duplicate groups by (student_id, exam_id, topic_id).
    2. Within each group, keep the row with the **highest id** (most
       recently inserted, which correlates with the most up-to-date
       progress).  All other rows in the group are deleted.

    This is safe because StudentTopicProgress rows are append-only
    per (student, exam, topic) — the latest row always has the most
    accurate streak/theta counters.  Evidence and generated-question
    rows reference ``adaptive_exams`` and ``generated_questions``, not
    ``student_topic_progress``, so deleting progress rows has no
    downstream FK impact.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '009'
down_revision: Union[str, None] = '008'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # Step 1: Delete duplicate rows, keeping the one with the highest id
    # per (student_id, exam_id, topic_id) group.
    conn.execute(sa.text(
        """
        DELETE FROM student_topic_progress
        WHERE id NOT IN (
            SELECT MAX(id)
            FROM student_topic_progress
            GROUP BY student_id, exam_id, topic_id
        )
        """
    ))

    # Step 2: Now safe to add the unique constraint.
    op.create_unique_constraint(
        "uq_student_exam_topic",
        "student_topic_progress",
        ["student_id", "exam_id", "topic_id"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_student_exam_topic", "student_topic_progress", type_="unique")

"""add current_question_started_at to adaptive_exams

Revision ID: 013
Revises: 012
Create Date: 2026-07-28 14:00:00.000000

Adds ``adaptive_exams.current_question_started_at`` — nullable
server-owned timestamp that records when the current question was
served.  Used to compute authoritative elapsed seconds on answer
submission so anomaly detection never trusts client-provided timing.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '013'
down_revision: Union[str, None] = '012'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "adaptive_exams",
        sa.Column(
            "current_question_started_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("adaptive_exams", "current_question_started_at")

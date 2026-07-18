"""add current_theta and consumed to student_topic_progress

Revision ID: 008
Revises: 55e16bed1497
Create Date: 2026-06-22 20:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '008'
down_revision: Union[str, None] = '55e16bed1497'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'student_topic_progress',
        sa.Column('current_theta', sa.Float(), nullable=False, server_default='0.0'),
    )
    op.add_column(
        'student_topic_progress',
        sa.Column('consumed', sa.Boolean(), nullable=False, server_default='false'),
    )


def downgrade() -> None:
    op.drop_column('student_topic_progress', 'consumed')
    op.drop_column('student_topic_progress', 'current_theta')

"""add anomaly detection columns to adaptive_exam_responses

Revision ID: 012
Revises: 011
Create Date: 2026-07-28 12:00:00.000000

Adds nullable columns for Isolation Forest anomaly detection results
on ``adaptive_exam_responses``:
- ``anomaly_flag`` (bool) — 1 = anomalous, 0 = normal
- ``anomaly_score`` (float) — continuous anomaly score
- ``predicted_class`` (str) — "Anomaly" or "Normal"
- ``response_interpretation`` (str) — human-readable label
- ``elapsed_seconds`` (float) — per-question elapsed time for traceability

All columns are nullable so existing rows are unaffected and the feature
can be rolled out incrementally.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '012'
down_revision: Union[str, None] = '011'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "adaptive_exam_responses",
        sa.Column("anomaly_flag", sa.Boolean, nullable=True),
    )
    op.add_column(
        "adaptive_exam_responses",
        sa.Column("anomaly_score", sa.Float, nullable=True),
    )
    op.add_column(
        "adaptive_exam_responses",
        sa.Column("predicted_class", sa.String(16), nullable=True),
    )
    op.add_column(
        "adaptive_exam_responses",
        sa.Column("response_interpretation", sa.String(64), nullable=True),
    )
    op.add_column(
        "adaptive_exam_responses",
        sa.Column("elapsed_seconds", sa.Float, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("adaptive_exam_responses", "elapsed_seconds")
    op.drop_column("adaptive_exam_responses", "response_interpretation")
    op.drop_column("adaptive_exam_responses", "predicted_class")
    op.drop_column("adaptive_exam_responses", "anomaly_score")
    op.drop_column("adaptive_exam_responses", "anomaly_flag")

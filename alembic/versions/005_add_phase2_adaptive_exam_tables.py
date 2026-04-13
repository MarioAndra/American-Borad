"""add phase2 adaptive exam tables and irt columns

Revision ID: 005
Revises: 004
Create Date: 2026-04-13
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("questions", sa.Column("irt_a", sa.Float(), nullable=True))
    op.add_column("questions", sa.Column("irt_b", sa.Float(), nullable=True))
    op.add_column("questions", sa.Column("irt_c", sa.Float(), nullable=True))

    op.create_table(
        "adaptive_exams",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("student_id", sa.Integer(), nullable=False),
        sa.Column("phase_id", sa.Integer(), nullable=False),
        sa.Column("status", postgresql.ENUM("Pending", "InProgress", "Completed", name="exam_status", create_type=False), nullable=False),
        sa.Column("max_questions", sa.Integer(), nullable=False),
        sa.Column("answered_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("current_theta", sa.Float(), server_default=sa.text("0"), nullable=False),
        sa.Column("score", sa.Numeric(5, 2), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("answered_count <= max_questions", name="ck_adaptive_exam_answered_max"),
        sa.ForeignKeyConstraint(["student_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["phase_id"], ["phases.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_adaptive_exams_student_id", "adaptive_exams", ["student_id"], unique=False)
    op.create_index("ix_adaptive_exams_phase_id", "adaptive_exams", ["phase_id"], unique=False)
    op.create_index("ix_adaptive_exams_status", "adaptive_exams", ["status"], unique=False)

    op.create_table(
        "adaptive_exam_responses",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("adaptive_exam_id", sa.Integer(), nullable=False),
        sa.Column("question_id", sa.Integer(), nullable=False),
        sa.Column("choice_id", sa.Integer(), nullable=False),
        sa.Column("order_index", sa.Integer(), nullable=False),
        sa.Column("is_correct", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("theta_before", sa.Float(), nullable=False),
        sa.Column("theta_after", sa.Float(), nullable=False),
        sa.Column("answered_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["adaptive_exam_id"], ["adaptive_exams.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["question_id"], ["questions.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["choice_id"], ["choices.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("adaptive_exam_id", "question_id", name="uq_adaptive_exam_question"),
    )
    op.create_index("ix_adaptive_exam_responses_adaptive_exam_id", "adaptive_exam_responses", ["adaptive_exam_id"], unique=False)
    op.create_index("ix_adaptive_exam_responses_question_id", "adaptive_exam_responses", ["question_id"], unique=False)
    op.create_index("ix_adaptive_exam_responses_choice_id", "adaptive_exam_responses", ["choice_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_adaptive_exam_responses_choice_id", table_name="adaptive_exam_responses")
    op.drop_index("ix_adaptive_exam_responses_question_id", table_name="adaptive_exam_responses")
    op.drop_index("ix_adaptive_exam_responses_adaptive_exam_id", table_name="adaptive_exam_responses")
    op.drop_table("adaptive_exam_responses")

    op.drop_index("ix_adaptive_exams_status", table_name="adaptive_exams")
    op.drop_index("ix_adaptive_exams_phase_id", table_name="adaptive_exams")
    op.drop_index("ix_adaptive_exams_student_id", table_name="adaptive_exams")
    op.drop_table("adaptive_exams")

    op.drop_column("questions", "irt_c")
    op.drop_column("questions", "irt_b")
    op.drop_column("questions", "irt_a")
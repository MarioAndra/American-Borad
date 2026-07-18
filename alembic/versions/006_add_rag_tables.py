"""add rag tables — knowledge_documents, knowledge_chunks, generated_questions and related

Revision ID: 006
Revises: 005
Create Date: 2026-05-20
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "knowledge_documents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("course_name", sa.String(255), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("topic_id", sa.Integer(), sa.ForeignKey("topics.id", ondelete="SET NULL"), nullable=True),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("resource_type", sa.String(50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_knowledge_documents_topic_id", "knowledge_documents", ["topic_id"], unique=False)

    op.create_table(
        "knowledge_chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("topic_id", sa.Integer(), sa.ForeignKey("topics.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_knowledge_chunks_document_id", "knowledge_chunks", ["document_id"], unique=False)
    op.create_index("ix_knowledge_chunks_topic_id", "knowledge_chunks", ["topic_id"], unique=False)

    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'generated_question_status') THEN
                CREATE TYPE generated_question_status AS ENUM ('draft','approved','rejected','auto_approved');
            END IF;
        END $$;
    """)

    op.create_table(
        "generated_questions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("topic_id", sa.Integer(), sa.ForeignKey("topics.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("source_exam_id", sa.Integer(), sa.ForeignKey("adaptive_exams.id", ondelete="SET NULL"), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("choices", postgresql.JSON(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column("difficulty_estimate", sa.Float(), nullable=True),
        sa.Column("status", postgresql.ENUM("draft", "approved", "rejected", "auto_approved", name="generated_question_status", create_type=False), nullable=False),
        sa.Column("review_required", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("validation_report", postgresql.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_generated_questions_topic_id", "generated_questions", ["topic_id"], unique=False)
    op.create_index("ix_generated_questions_status", "generated_questions", ["status"], unique=False)

    op.create_table(
        "generated_question_evidence",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("generated_question_id", sa.Integer(), sa.ForeignKey("generated_questions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_id", sa.Integer(), sa.ForeignKey("knowledge_chunks.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("relevance_score", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_evidence_generated_question_id", "generated_question_evidence", ["generated_question_id"], unique=False)
    op.create_index("ix_evidence_chunk_id", "generated_question_evidence", ["chunk_id"], unique=False)

    op.create_table(
        "generated_question_reviews",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("generated_question_id", sa.Integer(), sa.ForeignKey("generated_questions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("reviewer_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("decision", sa.String(20), nullable=False),
        sa.Column("comments", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_reviews_generated_question_id", "generated_question_reviews", ["generated_question_id"], unique=False)

    op.create_table(
        "student_topic_progress",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("student_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("exam_id", sa.Integer(), sa.ForeignKey("adaptive_exams.id", ondelete="CASCADE"), nullable=False),
        sa.Column("topic_id", sa.Integer(), sa.ForeignKey("topics.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("current_streak", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("questions_asked", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("generated_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("avg_theta", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_stp_student_id", "student_topic_progress", ["student_id"], unique=False)
    op.create_index("ix_stp_exam_id", "student_topic_progress", ["exam_id"], unique=False)
    op.create_index("ix_stp_topic_id", "student_topic_progress", ["topic_id"], unique=False)


def downgrade() -> None:
    op.drop_table("student_topic_progress")
    op.drop_table("generated_question_reviews")
    op.drop_table("generated_question_evidence")
    op.drop_table("generated_questions")
    op.execute("DROP TYPE IF EXISTS generated_question_status")
    op.drop_table("knowledge_chunks")
    op.drop_table("knowledge_documents")

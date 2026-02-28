"""full initial schema
Revision ID: 001
Revises: 
Create Date: 2026-02-18
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role') THEN
                CREATE TYPE user_role AS ENUM ('Student','Admin');
            END IF;
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'difficulty_level') THEN
                CREATE TYPE difficulty_level AS ENUM ('Easy','Medium','Hard');
            END IF;
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'exam_status') THEN
                CREATE TYPE exam_status AS ENUM ('Pending','InProgress','Completed');
            END IF;
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'cognitive_level') THEN
                CREATE TYPE cognitive_level AS ENUM ('Knowledge','Application','Analysis');
            END IF;
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'question_type') THEN
                CREATE TYPE question_type AS ENUM ('SingleChoice','MultipleSelect');
            END IF;
        END $$;
    """)

    op.create_table(
        'phases',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('name', name='uq_phases_name'),
    )

    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('full_name', sa.String(255), nullable=False),
        sa.Column('email', sa.String(320), nullable=False),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('role', postgresql.ENUM('Student','Admin', name='user_role', create_type=False), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column('is_verified', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('verification_token', sa.String(128), nullable=True),
        sa.Column('verification_sent_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('verified_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('email', name='uq_users_email'),
    )
    op.create_index('ix_users_verification_token', 'users', ['verification_token'], unique=False)

    op.create_table(
        'abet_criteria',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('code', sa.String(50), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('code', name='uq_abet_criteria_code'),
    )

    op.create_table(
        'topics',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('phase_id', sa.Integer(), sa.ForeignKey('phases.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('phase_id','name', name='uq_topics_phase_name'),
    )
    op.create_index('ix_topics_phase_id', 'topics', ['phase_id'], unique=False)
    op.create_index('ix_topics_name', 'topics', ['name'], unique=False)

    op.create_table(
        'subtopics',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('topic_id', sa.Integer(), sa.ForeignKey('topics.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('topic_id','name', name='uq_subtopics_topic_name'),
    )
    op.create_index('ix_subtopics_topic_id', 'subtopics', ['topic_id'], unique=False)
    op.create_index('ix_subtopics_name', 'subtopics', ['name'], unique=False)

    op.create_table(
        'questions',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('difficulty', postgresql.ENUM('Easy','Medium','Hard', name='difficulty_level', create_type=False), nullable=False),
        sa.Column('cognitive_level', postgresql.ENUM('Knowledge','Application','Analysis', name='cognitive_level', create_type=False), nullable=False),
        sa.Column('question_type', postgresql.ENUM('SingleChoice','MultipleSelect', name='question_type', create_type=False), nullable=False),
        sa.Column('subtopic_id', sa.Integer(), sa.ForeignKey('subtopics.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('abet_criterion_id', sa.Integer(), sa.ForeignKey('abet_criteria.id', ondelete='RESTRICT'), nullable=True),
        sa.Column('created_by', sa.Integer(), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column('explanation', sa.Text(), nullable=True),
        sa.Column('common_mistake', sa.Text(), nullable=True),
        sa.Column('skill_gap', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('text', name='uq_questions_text'),
    )
    op.create_index('ix_questions_difficulty', 'questions', ['difficulty'], unique=False)
    op.create_index('ix_questions_cognitive_level', 'questions', ['cognitive_level'], unique=False)
    op.create_index('ix_questions_question_type', 'questions', ['question_type'], unique=False)
    op.create_index('ix_questions_subtopic_id', 'questions', ['subtopic_id'], unique=False)
    op.create_index('ix_questions_abet_criterion_id', 'questions', ['abet_criterion_id'], unique=False)

    op.create_table(
        'choices',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('question_id', sa.Integer(), sa.ForeignKey('questions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('is_correct', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.UniqueConstraint('question_id', 'text', name='uq_choices_question_text'),
    )
    op.create_index('ix_choices_question_id', 'choices', ['question_id'], unique=False)

    op.create_table(
        'exams',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('student_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('phase_id', sa.Integer(), sa.ForeignKey('phases.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('total_questions', sa.Integer(), nullable=False, server_default=sa.text('100')),
        sa.Column('easy_count', sa.Integer(), nullable=False),
        sa.Column('medium_count', sa.Integer(), nullable=False),
        sa.Column('hard_count', sa.Integer(), nullable=False),
        sa.Column('status', postgresql.ENUM('Pending','InProgress','Completed', name='exam_status', create_type=False), nullable=False),
        sa.Column('score', sa.Numeric(5, 2), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('submitted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint('easy_count + medium_count + hard_count = total_questions', name='ck_exam_counts_total'),
    )
    op.create_index('ix_exams_student_id', 'exams', ['student_id'], unique=False)
    op.create_index('ix_exams_phase_id', 'exams', ['phase_id'], unique=False)
    op.create_index('ix_exams_status', 'exams', ['status'], unique=False)

    op.create_table(
        'exam_questions',
        sa.Column('exam_id', sa.Integer(), sa.ForeignKey('exams.id', ondelete='CASCADE'), nullable=False),
        sa.Column('question_id', sa.Integer(), sa.ForeignKey('questions.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('position', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('exam_id','question_id', name='pk_exam_questions'),
    )
    op.create_index('ix_exam_questions_exam_id', 'exam_questions', ['exam_id'], unique=False)
    op.create_index('ix_exam_questions_question_id', 'exam_questions', ['question_id'], unique=False)

    op.create_table(
        'student_answers',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('exam_id', sa.Integer(), sa.ForeignKey('exams.id', ondelete='CASCADE'), nullable=False),
        sa.Column('question_id', sa.Integer(), sa.ForeignKey('questions.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('choice_id', sa.Integer(), sa.ForeignKey('choices.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('is_correct', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('answered_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('exam_id','question_id','choice_id', name='uq_student_answers_sel'),
    )
    op.create_index('ix_student_answers_exam_id', 'student_answers', ['exam_id'], unique=False)
    op.create_index('ix_student_answers_question_id', 'student_answers', ['question_id'], unique=False)
    op.create_index('ix_student_answers_choice_id', 'student_answers', ['choice_id'], unique=False)

    op.create_table(
        'token_blacklist',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('jti', sa.String(255), nullable=False),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('jti', name='uq_token_blacklist_jti'),
    )
    op.create_index('ix_token_blacklist_user_id', 'token_blacklist', ['user_id'], unique=False)


def downgrade():
    op.drop_table('token_blacklist')
    op.drop_table('student_answers')
    op.drop_table('exam_questions')
    op.drop_table('exams')
    op.drop_table('choices')
    op.drop_table('questions')
    op.drop_table('subtopics')
    op.drop_table('topics')
    op.drop_table('abet_criteria')
    op.drop_table('users')
    op.drop_table('phases')
    op.execute("DROP TYPE IF EXISTS exam_status")
    op.execute("DROP TYPE IF EXISTS difficulty_level")
    op.execute("DROP TYPE IF EXISTS cognitive_level")
    op.execute("DROP TYPE IF EXISTS question_type")
    op.execute("DROP TYPE IF EXISTS user_role")
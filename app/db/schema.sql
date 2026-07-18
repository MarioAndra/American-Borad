# PostgreSQL schema blueprint (documentation only; no executable SQL here).
#
# Responsibility:
# - Describe the database objects to be created via migrations.
# - Serve as a reference for table structures, indexes, FKs, and enums.
#
# ENUMS:
# - difficulty_level: 'Easy', 'Medium', 'Hard'
# - exam_status: 'Pending', 'InProgress', 'Completed'
# - user_role: 'Student', 'Admin'
# - cognitive_level: 'Knowledge', 'Application', 'Analysis'
# - question_type: 'SingleChoice', 'MultipleSelect'
# - generated_question_status: 'draft', 'approved', 'rejected', 'auto_approved'
#
# TAXONOMY TABLES:
# TABLE: phases
# - id PK
# - name TEXT UNIQUE NOT NULL
# - description TEXT NULL
# - created_at TIMESTAMPTZ NOT NULL DEFAULT now()
# INDEXES: ON name
#
# TABLE: topics
# - id PK
# - phase_id FK -> phases(id) ON DELETE CASCADE
# - name TEXT NOT NULL
# - description TEXT NULL
# - created_at TIMESTAMPTZ NOT NULL DEFAULT now()
# UNIQUE: (phase_id, name)
# INDEXES: ON phase_id, name
#
# TABLE: subtopics
# - id PK
# - topic_id FK -> topics(id) ON DELETE CASCADE
# - name TEXT NOT NULL
# - description TEXT NULL
# - created_at TIMESTAMPTZ NOT NULL DEFAULT now()
# UNIQUE: (topic_id, name)
# INDEXES: ON topic_id, name
#
# TABLE: users
# - id PK
# - full_name TEXT NOT NULL
# - email TEXT UNIQUE NOT NULL
# - hashed_password TEXT NOT NULL
# - role user_role NOT NULL DEFAULT 'Student'
# - is_active BOOLEAN NOT NULL DEFAULT TRUE
# - created_at TIMESTAMPTZ NOT NULL DEFAULT now()
# - updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
# INDEXES: ON email
#
# TABLE: abet_criteria
# - id PK
# - code TEXT UNIQUE NOT NULL
# - name TEXT NOT NULL
# - description TEXT NULL
# - created_at TIMESTAMPTZ NOT NULL DEFAULT now()
# INDEXES: ON code
#
# TABLE: questions
# - id PK
# - text TEXT UNIQUE NOT NULL
# - difficulty difficulty_level NOT NULL
# - cognitive_level cognitive_level NOT NULL
# - question_type question_type NOT NULL
# - subtopic_id FK -> subtopics(id) ON DELETE RESTRICT
# - abet_criterion_id FK -> abet_criteria(id) ON DELETE RESTRICT
# - created_by FK -> users(id) ON DELETE SET NULL
# - is_active BOOLEAN NOT NULL DEFAULT TRUE
# - explanation TEXT NULL
# - common_mistake TEXT NULL
# - skill_gap TEXT NULL
# - created_at TIMESTAMPTZ NOT NULL DEFAULT now()
# INDEXES: ON difficulty, subtopic_id, cognitive_level, question_type, abet_criterion_id
#
# TABLE: choices
# - id PK
# - question_id FK -> questions(id) ON DELETE CASCADE
# - text TEXT NOT NULL
# - is_correct BOOLEAN NOT NULL DEFAULT FALSE
# UNIQUE: (question_id, text)
# INDEXES: ON question_id
#
# TABLE: exams
# - id PK
# - student_id FK -> users(id) ON DELETE CASCADE
# - phase_id FK -> phases(id) ON DELETE RESTRICT
# - total_questions INT NOT NULL DEFAULT 100
# - easy_count INT NOT NULL
# - medium_count INT NOT NULL
# - hard_count INT NOT NULL
# - status exam_status NOT NULL DEFAULT 'Pending'
# - score NUMERIC(5,2) NULL
# - started_at TIMESTAMPTZ NULL
# - submitted_at TIMESTAMPTZ NULL
# - created_at TIMESTAMPTZ NOT NULL DEFAULT now()
# INDEXES: ON student_id, phase_id, status
# CHECK: easy_count + medium_count + hard_count = total_questions
#
# TABLE: exam_questions
# - exam_id FK -> exams(id) ON DELETE CASCADE
# - question_id FK -> questions(id) ON DELETE RESTRICT
# - position INT NULL
# PRIMARY KEY: (exam_id, question_id)
# INDEXES: ON question_id
#
# TABLE: student_answers
# - id PK
# - exam_id FK -> exams(id) ON DELETE CASCADE
# - question_id FK -> questions(id) ON DELETE RESTRICT
# - choice_id FK -> choices(id) ON DELETE RESTRICT
# - is_correct BOOLEAN NOT NULL
# - answered_at TIMESTAMPTZ NOT NULL DEFAULT now()
# UNIQUE: (exam_id, question_id, choice_id)
# INDEXES: ON exam_id, question_id, choice_id
#
# TABLE: token_blacklist
# - id PK
# - jti TEXT UNIQUE NOT NULL
# - user_id FK -> users(id) ON DELETE CASCADE
# - expires_at TIMESTAMPTZ NOT NULL
# - created_at TIMESTAMPTZ NOT NULL DEFAULT now()
# INDEXES: ON user_id, expires_at
#
# RAG TABLES (Phase II Topic-Aware):
# EXTENSION: vector (pgvector)
#
# TABLE: knowledge_documents
# - id PK
# - course_name TEXT NOT NULL
# - title TEXT NOT NULL
# - topic_id FK -> topics(id) ON DELETE SET NULL
# - source_path TEXT NOT NULL
# - resource_type TEXT NOT NULL (page/pdf/transcript)
# - created_at TIMESTAMPTZ NOT NULL DEFAULT now()
# INDEXES: ON topic_id
#
# TABLE: knowledge_chunks
# - id PK
# - document_id FK -> knowledge_documents(id) ON DELETE CASCADE
# - chunk_index INT NOT NULL
# - text TEXT NOT NULL
# - embedding vector(1536) NULL
# - topic_id FK -> topics(id) ON DELETE SET NULL
# - created_at TIMESTAMPTZ NOT NULL DEFAULT now()
# INDEXES: ON document_id, topic_id
#
# TABLE: generated_questions
# - id PK
# - topic_id FK -> topics(id) ON DELETE RESTRICT
# - source_exam_id FK -> adaptive_exams(id) ON DELETE SET NULL
# - text TEXT NOT NULL
# - choices JSON NOT NULL (array of {text, is_correct})
# - explanation TEXT NOT NULL
# - difficulty_estimate FLOAT NULL
# - status generated_question_status NOT NULL
# - review_required BOOL NOT NULL DEFAULT TRUE
# - validation_report JSON NULL
# - created_at TIMESTAMPTZ NOT NULL DEFAULT now()
# INDEXES: ON topic_id, status
#
# TABLE: generated_question_evidence
# - id PK
# - generated_question_id FK -> generated_questions(id) ON DELETE CASCADE
# - chunk_id FK -> knowledge_chunks(id) ON DELETE RESTRICT
# - relevance_score FLOAT NULL
# INDEXES: ON generated_question_id, chunk_id
#
# TABLE: generated_question_reviews
# - id PK
# - generated_question_id FK -> generated_questions(id) ON DELETE CASCADE
# - reviewer_id FK -> users(id) ON DELETE SET NULL
# - decision TEXT NOT NULL
# - comments TEXT NULL
# - reviewed_at TIMESTAMPTZ NOT NULL DEFAULT now()
# INDEXES: ON generated_question_id
#
# TABLE: student_topic_progress
# - id PK
# - student_id FK -> users(id) ON DELETE CASCADE
# - exam_id FK -> adaptive_exams(id) ON DELETE CASCADE
# - topic_id FK -> topics(id) ON DELETE RESTRICT
# - current_streak INT NOT NULL DEFAULT 0
# - questions_asked INT NOT NULL DEFAULT 0
# - generated_count INT NOT NULL DEFAULT 0
# - avg_theta FLOAT NULL
# - created_at TIMESTAMPTZ NOT NULL DEFAULT now()
# INDEXES: ON student_id, exam_id, topic_id
#
# NOTES:
# - All timestamps use timezone-aware TIMESTAMPTZ.
# - Use deterministic naming conventions for constraints in Alembic.
# - Consider adding updated_at triggers or app-level updates on change.
# - For SingleChoice questions, enforce single selection at service level or via partial constraint.
# - For MultipleSelect questions, grading must compare selected set vs correct set.

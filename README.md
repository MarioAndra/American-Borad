# American‑Borad Backend

## Prerequisites

- Python 3.13
- PostgreSQL 16 or newer

## Setup Virtual Environment

### Windows

```powershell
# Create a virtual environment
python -m venv .venv

# Activate (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Command Prompt)
.\.venv\Scripts\activate

# Deactivate when finished
deactivate
```

### macOS / Linux

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Deactivate when finished
deactivate
```

## Installation

With your virtual environment activated:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Environment Variables

Create your environment file from the example and fill in the required values (database URL/credentials, secrets, etc.):

### Windows

```powershell
Copy-Item .env.example .env
```

or (Command Prompt):

```cmd
copy .env.example .env
```

### macOS / Linux

```bash
cp .env.example .env
```

Open `.env` and update values such as `DATABASE_URL` (or the corresponding `POSTGRES_*` variables) to point to your PostgreSQL instance.

## Database Migrations

Alembic is used for database migrations.

1) Ensure the database URL is available to Alembic via `ALEMBIC_DATABASE_URL` (preferred) or `DATABASE_URL`.

Windows (PowerShell):

```powershell
$env:ALEMBIC_DATABASE_URL = "postgresql+psycopg://<user>:<password>@localhost:5432/<dbname>"
alembic upgrade head
```

macOS / Linux:

```bash
export ALEMBIC_DATABASE_URL="postgresql+psycopg://<user>:<password>@localhost:5432/<dbname>"
alembic upgrade head
```

Useful commands:

```bash
# Create a new migration from models (optional)
alembic revision --autogenerate -m "Describe your change"

# Apply all migrations
alembic upgrade head

# Show current migration
alembic current
```

## Data Import

To import questions from Excel files located in the data directory:

```powershell
python -m app.scripts.import_questions_from_excel --dir "./app/data"

python -m app.scripts.import_questions_from_excel --dir "c:\Users\Mouneer\UniversityProjects\American-Borad\app\data\Phase II"
```

## Phase II Adaptive Exam Flow

Phase II is implemented as a separate flow and does not change Phase I endpoints/behavior.

### Configuration (`.env`)

Add/confirm:

```env
PHASE2_ENABLED=true
PHASE2_PHASE_ID=2
PHASE2_MAX_QUESTIONS=20
PHASE2_PASSING_SCORE=75
PHASE2_INITIAL_THETA=0.0
```

### Data Requirements (Eligibility)

For a question to be eligible in Phase II adaptive selection:

- It belongs to `PHASE2_PHASE_ID` through topic/subtopic relations.
- `questions.is_active = true`
- It has **exactly one** correct choice (`choices.is_correct = true`).

### API Endpoints

All adaptive endpoints are under `/api/v1/phase2`:

- `POST /api/v1/phase2/exams/start`
  - Body: `{ "phase_id": 2 }` (optional; defaults to `PHASE2_PHASE_ID`).
  - Starts an adaptive exam and returns the first question.
- `POST /api/v1/phase2/exams/{exam_id}/answer`
  - Body: `{ "question_id": <id>, "choice_id": <id> }`
  - Submits one answer and returns next question/progress.
- `GET /api/v1/phase2/exams/{exam_id}`
  - Returns current progress or final result if completed.

### Suggested Test Sequence

1. Login and get student `access_token`.
2. Start exam using `POST /api/v1/phase2/exams/start`.
3. Read `exam_id` and `current_question` from response.
4. Submit answers one-by-one using `/answer` until `status = Completed`.
5. Fetch final state/result with `GET /api/v1/phase2/exams/{exam_id}`.

### Quick Eligibility Check SQL

```sql
SELECT COUNT(*) AS eligible_count
FROM questions q
JOIN subtopics st ON st.id = q.subtopic_id
JOIN topics t ON t.id = st.topic_id
JOIN (
  SELECT question_id, COUNT(*) AS correct_cnt
  FROM choices
  WHERE is_correct = true
  GROUP BY question_id
) cc ON cc.question_id = q.id
WHERE t.phase_id = 2
  AND q.is_active = true
  AND cc.correct_cnt = 1;
```

## Using Uvicorn (FastAPI/Starlette)
 ``uvicorn app.main:app --reload``

1. Wipe the DB volume (this destroys the data, not the image):
docker compose down -v
2. Restart everything (migrations auto-run on first start):
docker compose up -d
3. Import Phase I Excel (inside the running container):
docker compose exec app python -m app.scripts.import_questions_from_excel --dir "/app/data"
4. Re-ingest courses for RAG via the API:
# Seed an admin first
docker compose exec app python -m app.scripts.seed_admin --email admin@example.com --password admin123




# Create an admin user
python -m app.scripts.seed_admin --email admin@example.com --password admin123
# Full name override
python -m app.scripts.seed_admin --email admin@example.com --password admin123 --name "Super Admin"
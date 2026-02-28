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
```

## Using Uvicorn (FastAPI/Starlette)
 ``uvicorn app.main:app --reload``

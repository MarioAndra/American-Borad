# AGENTS.md — American Board of AI Exam System

Python 3.13, FastAPI, PostgreSQL 16+, SQLAlchemy 2.0 `Mapped`/`mapped_column`, Pydantic v2.

## Commands

| Action | Command |
|--------|---------|
| Dev server | `uvicorn app.main:app --reload` (port 8000) |
| Migration (apply) | `alembic upgrade head` — reads `ALEMBIC_DATABASE_URL` env var first, falls back to `DATABASE_URL` from `.env`. `alembic.ini` `sqlalchemy.url` is **intentionally empty**. |
| Migration (create) | `alembic revision --autogenerate -m "msg"` — models auto-imported via `alembic/env.py` line `import app.models` |
| Import Excel | `python -m app.scripts.import_questions_from_excel --dir "./app/data"` |
| Seed admin | `python -m app.scripts.seed_admin --email X --password Y [--name "Z"]` |
| Docker | `docker compose up` — spins up postgres:17, weaviate, and the app (migrations auto-run on start via `CMD` in Dockerfile) |
| Docker hot-reload | `docker compose cp app/services/foo.py app:/app/app/services/foo.py && docker compose restart app` (no rebuild) |
| Wipe DB | `docker compose down -v` (destroys volumes) |
| Test | No test framework configured. Use `docker compose exec app python -c "..."` or write inline tests. |

## Architecture

- **FastAPI** at `app/main.py`, routes under `/api/v1`. CORS allows `localhost:5173` (Vite frontend).
- **Two exam modes**: Phase I (fixed-format exam), Phase II (adaptive IRT via catsim 3PL + MaxInfoSelector).
- **Registered routers** (`app/api/v1/__init__.py`): `auth` (prefix `/auth`), `exams`, `adaptive_exams` (prefix `/phase2`), `rag` (prefix `/rag`). The endpoint files `imports.py`, `phases.py`, `questions.py`, `subtopics.py`, `topics.py`, `users.py` exist but are **not wired** into the router.
- **Phase II endpoints**: `POST /phase2/exams/start`, `POST /phase2/exams/{id}/answer`, `GET /phase2/exams/{id}`.
- **Phase II eligibility**: `questions.is_active = true`, exactly 1 correct choice, belongs to `phase_id = 2` (config `PHASE2_PHASE_ID`).
- **RAG (Phase II topic-aware generation)**: Triggered after `PHASE2_SUBTOPIC_BASE_QUESTION_COUNT` (default 4) same-topic streak. Dual vector store: pgvector + Weaviate. Providers: OpenAI, Gemini, or Groq (`RAG_EMBEDDING_PROVIDER` / `RAG_GENERATION_PROVIDER`). Groq uses OpenAI-compatible API at `GROQ_BASE_URL`.
- **Phase II two-phase ingest**: `POST /api/v1/rag/extract` (Phase 1 — PDF→chunks, no tokens) → `POST /api/v1/rag/embed` (Phase 2 — chunk→embedding, consumes tokens). Combined endpoint: `POST /api/v1/rag/ingest`. Three more admin endpoints: `POST /api/v1/rag/check` (readiness + optional smoke test), `POST /api/v1/rag/verify` (PG vs Weaviate reconciliation), `GET /api/v1/rag/questions` (list generated Qs), `PATCH /api/v1/rag/questions/{id}/review` (approve/reject).
- **IRT defaults** (auto-seeded if null): `a=1.0`, `c=0.2`, `b` from difficulty (Easy→-1.0, Medium→0.0, Hard→1.0).
- **Auth**: JWT HS256 (config default 15m access, `.env` overrides to 60m via `ACCESS_TOKEN_EXPIRE_MINUTES`; 30d refresh), jti-based token blacklist, SHA-256→bcrypt (12 rounds) password hashing, `password_changed_at` invalidates existing tokens.
- **Postman collection** at root `American.postman_collection.json`.

## RAG Pipeline — Key Constraints

- **Explicit `COURSE_TOPIC_NAMES` dict** in `rag_ingestion_service.py` maps course folder → Phase II topic name. Unmapped courses are rejected upfront. Add new courses here.
- **Force re-ingest** (`"force": true`): delete + re-extract. Deletes evidence rows referencing old chunk IDs first (`GeneratedQuestionEvidence.chunk_id` has `ondelete="RESTRICT"`). Generated questions survive the cascade.
- **`embedding_status` flow**: `pending` (after extract) → `completed` (all chunks in Weaviate) or `failed` (zero chunks or embed failure). The embed pipeline queries both `pending` and `failed` — retry is automatic.
- **Config-gated fallback**: `RAG_ALLOW_VECTOR_FALLBACK=false` (default). When `false`, retrieval returns `[]` on Weaviate failure instead of random PG chunks. Set `true` in `.env` for dev.
- **Weaviate client degrades gracefully** when unavailable. Weaviate runs on port 8080 in Docker.
- **Evidence rows** (`GeneratedQuestionEvidence`) are persisted for every retrieved chunk after generation — enables tracing which chunks informed each question.
- **Zero-chunk docs** are marked `failed` (not `completed`) and surfaced in readiness check + ingest response.
- **OCR quarantine**: PDFs with empty or near-empty extracted text (`< 50 chars`) are quarantined with a message suggesting `RAG_OCR_ENABLED`. Check `IngestResponse.quarantined`.

## Conventions

- `from __future__ import annotations` at top of every file.
- SQLAlchemy 2.0 `Mapped`/`mapped_column`, no legacy `Column` syntax.
- Pydantic v2 `BaseModel` for schemas.
- `.env` is gitignored; copy `.env.example`. `JWT_SECRET_KEY` needs `openssl rand -hex 32`.
- `courses/` dir holds RAG source PDFs (not exam content) — never delete or reorganize.
- Preserve `guide.md`, `Implementation plan.md`, `PROJECT_MAP.md`, all migration files.
- `prompt.md` references `planning-protocol.md`, `excution-engine.md`, `surgical-editing-protocol.md` — these are verbose safety rules for the session agent, not runtime configuration.

## Verify Changes

1. If migration changed: `alembic upgrade head`
2. If service code changed: `docker compose cp` + restart container
3. Spot-check the affected endpoint via curl or Postman collection

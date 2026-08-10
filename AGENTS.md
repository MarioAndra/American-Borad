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
| LangGraph tests | `docker compose exec app python -m app.scripts.test_langgraph_phase1` (86 tests, mocked LLM/Weaviate) |
| Run single test | `docker compose exec app python -c "from app.scripts.test_langgraph_phase1 import <func>; <func>()"` |

## Architecture

- **FastAPI** at `app/main.py`, routes under `/api/v1`. CORS allows `localhost:5173` (Vite frontend).
- **Two exam modes**: Phase I (fixed-format exam), Phase II (adaptive IRT via catsim 3PL + MaxInfoSelector).
- **Registered routers** (`app/api/v1/__init__.py`): `auth` (prefix `/auth`), `exams`, `adaptive_exams` (prefix `/phase2`), `rag` (prefix `/rag`). The endpoint files `imports.py`, `phases.py`, `questions.py`, `subtopics.py`, `topics.py`, `users.py` exist but are **not wired** into the router.
- **Phase II endpoints**: `POST /phase2/exams/start`, `POST /phase2/exams/{id}/answer`, `GET /phase2/exams/{id}`.
- **Phase II eligibility**: `questions.is_active = true`, exactly 1 correct choice, belongs to `phase_id = 2` (config `PHASE2_PHASE_ID`).
- **RAG (Phase II topic-aware generation)**: Triggered after `PHASE2_SUBTOPIC_BASE_QUESTION_COUNT` (default 4) same-topic streak. Dual vector store: pgvector + Weaviate. Providers: OpenAI, Gemini, or Groq (`RAG_EMBEDDING_PROVIDER` / `RAG_GENERATION_PROVIDER`). Groq uses OpenAI-compatible API at `GROQ_BASE_URL`.
- **Phase II two-phase ingest**: `POST /api/v1/rag/extract` (Phase 1 — PDF→chunks, no tokens) → `POST /api/v1/rag/embed` (Phase 2 — chunk→embedding, consumes tokens). Combined endpoint: `POST /api/v1/rag/ingest`. Three more admin endpoints: `POST /api/v1/rag/check` (readiness + optional smoke test), `POST /api/v1/rag/verify` (PG vs Weaviate reconciliation), `GET /api/v1/rag/questions` (list generated Qs), `PATCH /api/v1/rag/questions/{id}/review` (approve/reject).
- **LangGraph orchestration** (feature-flagged via `RAG_LANGGRAPH_ENABLED=false`): When enabled, `_try_generate_next()` delegates to `app.services.langgraph_rag_workflow.run_rag_graph` — a 19-node StateGraph: `gate_check → query_planner → retrieve → reranker → context_compressor → evidence_gate → retrieval_repair → generate → difficulty_estimator → grounding_validator → validate → distractor_validator → difficulty_calibrator → repair_decision → question_repair → duplicate_gate → confidence_gate → artifact_validator → persist`. On insufficient evidence, the graph retries once via `retrieval_repair` (broader queries) before aborting. When disabled (default), the legacy inline path in `_try_generate_next()` runs the same steps sequentially without LangGraph.
- **Query planner** (`app/services/query_planning_service.py`): Deterministic, template-based — generates 2–4 candidate retrieval queries by difficulty band + misconception angle. No LLM call. Planner failure falls back to empty list; retrieve node then uses legacy single-query path (`streak_info.topic_name`).
- **Multi-query retrieval** (`rag_retrieval_service.py`): When ≥2 planned queries, calls `retrieve_multi()` — batch-embeds all queries, searches Weaviate per embedding, merges by `chunk_id` keeping best similarity. Single query uses legacy `retrieve()`.
- **Reranker** (`retrieval_rerank_service.py`): Deterministic lexical-overlap + similarity fusion. Score = 0.6×similarity + 0.4×max_lexical_overlap across all candidate queries. Best-effort: exception or empty input falls back to original chunk order. No LLM call.
- **Context compressor** (`context_compression_service.py`): Deterministic post-reranker dedup + pruning. Deduplicates chunks via Jaccard text similarity (threshold 0.85), drops low-similarity chunks (floor 0.05), caps at 5 chunks. Best-effort: exception falls back to original chunk list. No LLM call.
- **Evidence gate** (`evidence_validation_service.py`): Deterministic evidence sufficiency check after compression. Requires: ≥1 chunk, avg similarity ≥ 0.15, at least one chunk with similarity ≥ 0.5. If insufficient, sets `failure_reason` and routes to `retrieval_repair` for one retry (or aborts if retries exhausted). Exception blocks generation as a safety fallback. No LLM call.
- **Retrieval repair** (`retrieval_repair_service.py`): Deterministic query broadening when evidence is insufficient. Generates simpler/broader queries (bare topic name, basic fundamentals, key principles) excluding those already tried. Bounded to 1 retry via `MAX_RETRIES`. If still insufficient after repair, graph aborts cleanly. No LLM call.
- **Grounding validator** (`grounding_validation_service.py`): Deterministic grounding check after question generation, before structural validation. Validates that the correct answer text and explanation have sufficient lexical support from retrieved evidence. Returns `GroundingReport` with `grounded`, `question_supported`, `answer_supported`, `explanation_supported`, `support_score`, and `issues`. Blocks generation when the question appears unsupported — fail-closed. Exception-safe: on failure, blocks as a safety fallback. No LLM call.
- **Distractor validator** (`distractor_validation_service.py`): Deterministic distractor quality check after structural validation, before duplicate gate. Validates that distractors are non-empty, distinct from each other (Jaccard < 0.85), and sufficiently separated from the correct answer (Jaccard < 0.70). Stop words are excluded during tokenisation (shared set with grounding validator). Returns `DistractorReport` with `valid`, `distinct_distractors`, `separated_from_correct`, `meaningful_distractors`, and `issues`. Empty/missing correct answer is a hard failure. Blocks generation when distractors are weak — fail-closed. Exception-safe: on failure, blocks as a safety fallback. No LLM call.
- **Duplicate gate** (`question_dedup_service.py`): Deterministic duplicate detection after validate, before persist. Compares generated question text against existing `GeneratedQuestion` rows (same topic, Jaccard ≥ 0.65) and the fixed question bank (same topic scope via SubTopic join, Jaccard ≥ 0.70). Blocks persistence when a duplicate is detected. Exception blocks persistence as a safety fallback. No LLM call.
- **Confidence gate** (`question_confidence_service.py`): Deterministic confidence-routing after duplicate gate, before persist. Scores 0–100 based on evidence quality (avg similarity, high-quality chunk count), retrieval repair usage, validation near-duplicate risk, and question/explanation completeness. Routes to `auto_approve` (≥70), `human_review` (40–69), or `reject` (<40). `RAG_REVIEW_REQUIRED=true` globally overrides `auto_approve` to `human_review` at persist time. Exception-safe: on failure, routes to `human_review` as a conservative safety fallback. Both raw and effective confidence routes are stored in `validation_report` JSON (`confidence_route_raw`, `confidence_route_effective`, `confidence_score`). No LLM call.
- **LangGraph telemetry** (`rag_telemetry_service.py`): `build_langgraph_trace(state, settings)` runs once at persist time. Stores a `langgraph_trace` sub-dict inside the existing `validation_report` JSON column on `GeneratedQuestion` — no schema migration needed. The trace captures `trace_id` (UUID), `retry_count`, `evidence` (chunk count, avg similarity, high-quality count), `confidence` (raw/effective route, score, reasons), and `validation` (schema_ok, issues). The admin `GET /rag/questions` endpoint already exposes `validation_report`, so traces are automatically visible. Exception-safe — if trace building fails, the question persists without it.
- **IRT defaults** (auto-seeded if null): `a=1.0`, `c=0.2`, `b` from difficulty (Easy→-1.0, Medium→0.0, Hard→1.0).
- **Auth**: JWT HS256 (config default 15m access, `.env` overrides to 60m via `ACCESS_TOKEN_EXPIRE_MINUTES`; 30d refresh), jti-based token blacklist, SHA-256→bcrypt (12 rounds) password hashing, `password_changed_at` invalidates existing tokens.
- **Postman collection** at root `American.postman_collection.json` (gitignored — may not be present).

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
- DB naming convention (`app/db/base.py`): `ix_`, `uq_`, `ck_`, `fk_`, `pk_` prefixes — Alembic autogenerate uses these.
- `.env` is gitignored; copy `.env.example`. `JWT_SECRET_KEY` needs `openssl rand -hex 32`.
- `courses/` dir holds RAG source PDFs (not exam content) — never delete or reorganize.
- Preserve `guide.md`, `Implementation plan.md`, all migration files.
- `prompt.md` references `planning-protocol.md`, `excution-engine.md`, `surgical-editing-protocol.md` — these are verbose safety rules for the session agent, not runtime configuration.

## Verify Changes

1. If migration changed: `alembic upgrade head`
2. If service code changed: `docker compose cp` + restart container
3. **Important**: `langgraph_rag_workflow.py` caches the compiled graph in `_compiled_graph`. Any graph structure change (new node, new edge, modified routing) **must** be followed by an app restart so the running API process does not keep using a stale compiled graph. Hot-copy and restart before running tests.
4. Run LangGraph tests to verify RAG flow: `docker compose exec app python -m app.scripts.test_langgraph_phase1`
5. Spot-check the affected endpoint via curl or Postman collection

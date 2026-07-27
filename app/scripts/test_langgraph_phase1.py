"""LangGraph Phase 1 + 2.1–2.8 — integration verification script.

Verifies that the feature-flagged LangGraph RAG flow is behaviourally
equivalent to the legacy inline flow in ``_try_generate_next()``,
that the Phase 2.1 query planner works correctly, that the
Phase 2.2 reranker node is correctly wired into the graph, and that
the Phase 2.8 grounding validator correctly blocks unsupported questions.

Run:
    docker compose exec app python -m app.scripts.test_langgraph_phase1

Setup assumptions:
    - Database is migrated (``alembic upgrade head``).
    - External services (LLM, Weaviate) are mocked — no API keys needed.
    - Tests run inside a transaction that is rolled back after each case,
      so no persistent side-effects are left in the database.
"""

from __future__ import annotations

import sys
import uuid
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from app.db.session import SessionLocal
from app.models import (
    AdaptiveExam,
    AdaptiveExamResponse,
    Choice,
    ExamStatus,
    Phase,
    Question,
    StudentTopicProgress,
    SubTopic,
    Topic,
    User,
    UserRole,
)
from app.models.rag import (
    GeneratedQuestion,
    GeneratedQuestionEvidence,
    KnowledgeChunk,
    KnowledgeDocument,
)
from app.services.adaptive_exam_service import _try_generate_next, submit_adaptive_answer, get_adaptive_exam
from app.services.generated_question_service import GenerationOutput
from app.services.generated_question_validation_service import ValidationReport, GeneratedQuestionValidationService
from app.services.query_planning_service import QueryPlannerService, QueryPlan
from app.services.rag_retrieval_service import RetrievedChunk
from app.services.retrieval_rerank_service import RerankerService, lexical_overlap
from app.services.context_compression_service import ContextCompressionService, text_similarity
from app.services.evidence_validation_service import EvidenceValidationService, EvidenceReport
from app.services.retrieval_repair_service import RetrievalRepairService
from app.services.question_dedup_service import QuestionDedupService, DedupReport
from app.services.grounding_validation_service import GroundingValidationService, GroundingReport
from app.services.distractor_validation_service import DistractorValidationService, DistractorReport
from app.services.difficulty_calibration_service import DifficultyCalibrationService, DifficultyCalibrationReport
from app.services.question_repair_service import QuestionRepairService, RepairReport
from app.services.question_confidence_service import QuestionConfidenceService, ConfidenceReport
from app.services.generated_artifact_validation_service import (
    GeneratedArtifactValidationService,
    ArtifactValidationReport,
)
from app.services.langgraph_rag_workflow import retrieval_repair, question_repair
from app.services.langgraph_rag_state import RAGGraphState
from app.services.topic_streak_service import StreakInfo


# ---------------------------------------------------------------------------
# Lightweight settings stand-in (avoids hitting .env)
# ---------------------------------------------------------------------------

class _Settings:
    def __init__(self, **kw: object) -> None:
        self.RAG_ENABLED: bool = bool(kw.get("RAG_ENABLED", True))
        self.RAG_LANGGRAPH_ENABLED: bool = bool(kw.get("RAG_LANGGRAPH_ENABLED", False))
        self.RAG_REVIEW_REQUIRED: bool = bool(kw.get("RAG_REVIEW_REQUIRED", False))
        self.GENERATED_MCQ_OPTION_COUNT: int = int(kw.get("GENERATED_MCQ_OPTION_COUNT", 4))
        self.GENERATED_MCQ_REQUIRE_CITATIONS: bool = bool(kw.get("GENERATED_MCQ_REQUIRE_CITATIONS", True))
        self.GENERATED_MCQ_REQUIRE_DISTRACTOR_RATIONALE: bool = bool(kw.get("GENERATED_MCQ_REQUIRE_DISTRACTOR_RATIONALE", True))
        self.PHASE2_SUBTOPIC_BASE_QUESTION_COUNT: int = int(kw.get("PHASE2_SUBTOPIC_BASE_QUESTION_COUNT", 4))
        self.PHASE2_SUBTOPIC_GENERATED_QUESTION_COUNT: int = int(kw.get("PHASE2_SUBTOPIC_GENERATED_QUESTION_COUNT", 1))
        self.RAG_GENERATION_PROVIDER: str = str(kw.get("RAG_GENERATION_PROVIDER", "openai"))
        self.PHASE2_PASSING_SCORE: float = float(kw.get("PHASE2_PASSING_SCORE", 70.0))


# ---------------------------------------------------------------------------
# Mock data factories
# ---------------------------------------------------------------------------

def _fake_chunks(
    topic_id: int,
    n: int = 3,
    *,
    chunk_ids: list[int] | None = None,
    document_id: int = 0,
) -> list[RetrievedChunk]:
    if chunk_ids is None:
        chunk_ids = [9_000_000 + i for i in range(n)]
    return [
        RetrievedChunk(
            chunk_id=cid,
            document_id=document_id,
            course_name="__lg_test__",
            title="Test Doc",
            text=f"Chunk {i}. Encryption protects confidentiality.",
            chunk_index=i,
            topic_id=topic_id,
            similarity=round(0.9 - i * 0.05, 4),
        )
        for i, cid in enumerate(chunk_ids)
    ]


def _fake_gen() -> GenerationOutput:
    return GenerationOutput(
        question_text="What is the primary purpose of encryption?",
        options=[
            {"text": "Protect confidentiality", "is_correct": True},
            {"text": "Increase speed", "is_correct": False},
            {"text": "Compress files", "is_correct": False},
            {"text": "Generate randomness", "is_correct": False},
        ],
        explanation="Encryption converts plaintext to ciphertext.",
        difficulty_estimate=0.4,
    )


def _fake_gen_with_artifacts() -> GenerationOutput:
    return GenerationOutput(
        question_text="What is the primary purpose of encryption?",
        options=[
            {"text": "Protect confidentiality", "is_correct": True},
            {"text": "Increase speed", "is_correct": False},
            {"text": "Compress files", "is_correct": False},
            {"text": "Generate randomness", "is_correct": False},
        ],
        explanation="Encryption converts plaintext to ciphertext.",
        difficulty_estimate=0.4,
        evidence_citations=[
            "Encryption converts plaintext to ciphertext for protection",
            "Chunk 1. Encryption protects confidentiality.",
        ],
        distractor_rationale={
            "1": "Increasing speed is unrelated to encryption purpose.",
            "2": "Compressing files is a separate process.",
            "3": "Randomness generation is used in key creation, not encryption itself.",
        },
    )


def _valid_report() -> ValidationReport:
    return ValidationReport(
        valid=True, issues=[], schema_ok=True,
        single_correct=True, non_duplicate=True, max_similarity=0.25,
    )


def _make_fake_gq(db, exam_id: int, topic_id: int) -> GeneratedQuestion:
    """Create a real GeneratedQuestion row in the DB for orchestration tests."""
    from app.models.enums import GeneratedQuestionStatus
    gq = GeneratedQuestion(
        topic_id=topic_id,
        source_exam_id=exam_id,
        text="What is the primary purpose of encryption?",
        choices=[
            {"text": "Protect confidentiality", "is_correct": True},
            {"text": "Increase speed", "is_correct": False},
            {"text": "Compress files", "is_correct": False},
            {"text": "Generate randomness", "is_correct": False},
        ],
        explanation="Encryption converts plaintext to ciphertext.",
        difficulty_estimate=0.4,
        status=GeneratedQuestionStatus.auto_approved,
        review_required=False,
    )
    db.add(gq)
    db.flush()
    return gq


def _invalid_report() -> ValidationReport:
    return ValidationReport(
        valid=False,
        issues=["Duplicate (Jaccard=0.92)"],
        schema_ok=True,
        single_correct=True,
        non_duplicate=False,
        max_similarity=0.92,
    )


# ---------------------------------------------------------------------------
# DB fixtures
# ---------------------------------------------------------------------------

def _setup(db, *, num_chunks: int = 3) -> dict[str, object]:
    """Create minimal test data.  Returns dict of created IDs + chunk info.

    Every call generates unique names/emails via a short UUID tag so
    repeated calls within the same process never collide on unique
    constraints, even if a prior rollback was incomplete.
    """
    tag = uuid.uuid4().hex[:8]

    student = User(
        full_name="LG Test Student",
        email=f"__lg_{tag}@example.com",
        hashed_password="x",
        role=UserRole.Student,
        is_active=True,
        is_verified=True,
    )
    db.add(student)
    db.flush()

    phase = Phase(name=f"__LG_Phase_{tag}__", description="test")
    db.add(phase)
    db.flush()

    topic = Topic(phase_id=phase.id, name=f"__LG_Topic_{tag}__", description="test")
    db.add(topic)
    db.flush()

    sub = SubTopic(topic_id=topic.id, name=f"__LG_Sub_{tag}__", description="test")
    db.add(sub)
    db.flush()

    exam = AdaptiveExam(
        student_id=student.id,
        phase_id=phase.id,
        status=ExamStatus.InProgress,
        max_questions=20,
        answered_count=0,
        current_theta=0.0,
    )
    db.add(exam)
    db.flush()

    # KnowledgeDocument + KnowledgeChunk so evidence FKs are satisfied
    doc = KnowledgeDocument(
        course_name="__lg_test__",
        title="Test Doc",
        source_path="/dev/null",
        resource_type="test",
        embedding_status="completed",
    )
    db.add(doc)
    db.flush()

    chunk_ids: list[int] = []
    for i in range(num_chunks):
        chunk = KnowledgeChunk(
            document_id=doc.id,
            chunk_index=i,
            text=f"Chunk {i}. Encryption protects confidentiality.",
            topic_id=topic.id,
        )
        db.add(chunk)
        db.flush()
        chunk_ids.append(chunk.id)

    return {
        "student_id": student.id,
        "phase_id": phase.id,
        "topic_id": topic.id,
        "subtopic_id": sub.id,
        "exam_id": exam.id,
        "document_id": doc.id,
        "chunk_ids": chunk_ids,
    }


def _set_streak(db, ids: dict[str, int], streak: int, generated: int = 0) -> None:
    p = StudentTopicProgress(
        student_id=ids["student_id"],
        exam_id=ids["exam_id"],
        topic_id=ids["topic_id"],
        current_streak=streak,
        questions_asked=streak,
        generated_count=generated,
        avg_theta=0.0,
    )
    db.add(p)
    db.flush()


# ---------------------------------------------------------------------------
# Mock invocation helper
# ---------------------------------------------------------------------------

def _invoke(
    db,
    ids: dict[str, int],
    settings: _Settings,
    *,
    chunks: list[RetrievedChunk] | None = None,
    gen: GenerationOutput | None = None,
    report: ValidationReport | None = None,
    langgraph: bool = False,
    planned_queries: list[str] | None = None,
    planner_side_effect: BaseException | None = None,
    return_mocks: bool = False,
    reranker_side_effect: BaseException | None = None,
    compressor_side_effect: BaseException | None = None,
    repair_queries: list[str] | None = None,
    repair_side_effect: BaseException | None = None,
    dedup_return: DedupReport | None = None,
    use_real_dedup: bool = False,
    confidence_return: ConfidenceReport | None = None,
    confidence_side_effect: BaseException | None = None,
    grounding_return: GroundingReport | None = None,
    grounding_side_effect: BaseException | None = None,
    distractor_return: DistractorReport | None = None,
    distractor_side_effect: BaseException | None = None,
    difficulty_return: DifficultyCalibrationReport | None = None,
    difficulty_side_effect: BaseException | None = None,
    question_repair_return: RepairReport | None = None,
    question_repair_side_effect: list | BaseException | None = None,
    artifact_return: ArtifactValidationReport | None = None,
    artifact_side_effect: BaseException | None = None,
):
    """Patch external services and call ``_try_generate_next``.

    When *return_mocks* is ``True`` the return value is a 13-tuple
    ``(result, mock_retrieval, mock_planner, mock_reranker, mock_gen_svc, mock_compressor, mock_repair, mock_dedup, mock_grounding, mock_distractor, mock_difficulty, mock_question_repair, mock_artifact)``
    so callers can assert on which services were called and with what
    arguments.
    """
    mock_ret = MagicMock()
    mock_ret.retrieve.return_value = chunks if chunks is not None else []
    mock_ret.retrieve_multi.return_value = chunks if chunks is not None else []
    mock_ret.close.return_value = None

    mock_gen_svc = MagicMock()
    mock_gen_svc.generate.return_value = gen

    mock_val_svc = MagicMock()
    mock_val_svc.validate.return_value = report

    # Default planned queries for LangGraph path
    if planned_queries is None:
        planned_queries = ["Cryptography", "common mistakes in Cryptography"]

    mock_planner = MagicMock()
    mock_plan = MagicMock()
    mock_plan.queries = planned_queries
    mock_planner.plan.return_value = mock_plan
    if planner_side_effect is not None:
        mock_planner.plan.side_effect = planner_side_effect

    # Reranker mock — by default returns chunks unchanged (identity)
    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = lambda c, q, **kw: list(c)
    if reranker_side_effect is not None:
        mock_reranker.rerank.side_effect = reranker_side_effect

    # Compressor mock — by default returns chunks unchanged (identity)
    mock_compressor = MagicMock()
    mock_compressor.compress.side_effect = lambda c, **kw: list(c)
    if compressor_side_effect is not None:
        mock_compressor.compress.side_effect = compressor_side_effect

    # Repair mock — by default returns a simple list of broader queries
    mock_repair = MagicMock()
    if repair_side_effect is not None:
        mock_repair.repair.side_effect = repair_side_effect
    else:
        mock_repair.repair.return_value = repair_queries or ["repaired query"]

    # Dedup mock — by default passes (no duplicate)
    mock_dedup = MagicMock()
    if dedup_return is not None:
        mock_dedup.check.return_value = dedup_return
    else:
        mock_dedup.check.return_value = DedupReport(
            is_duplicate=False, max_similarity=0.0,
            source="none", compared_count=0, threshold=0.65,
        )

    # Confidence mock — by default auto-approve with high score
    mock_confidence = MagicMock()
    if confidence_side_effect is not None:
        mock_confidence.evaluate.side_effect = confidence_side_effect
    elif confidence_return is not None:
        mock_confidence.evaluate.return_value = confidence_return
    else:
        mock_confidence.evaluate.return_value = ConfidenceReport(
            route="auto_approve", score=95.0, reasons=[],
        )

    # Grounding mock — by default passes (grounded)
    mock_grounding = MagicMock()
    if grounding_side_effect is not None:
        mock_grounding.validate.side_effect = grounding_side_effect
    elif grounding_return is not None:
        mock_grounding.validate.return_value = grounding_return
    else:
        mock_grounding.validate.return_value = GroundingReport(
            grounded=True, question_supported=True,
            answer_supported=True, explanation_supported=True,
            support_score=0.85, issues=[],
        )

    # Distractor mock — by default passes (valid distractors)
    mock_distractor = MagicMock()
    if distractor_side_effect is not None:
        mock_distractor.validate.side_effect = distractor_side_effect
    elif distractor_return is not None:
        mock_distractor.validate.return_value = distractor_return
    else:
        mock_distractor.validate.return_value = DistractorReport(
            valid=True, distinct_distractors=True,
            separated_from_correct=True, meaningful_distractors=True,
            issues=[],
        )

    # Difficulty calibrator mock — by default passes (aligned)
    mock_difficulty = MagicMock()
    if difficulty_side_effect is not None:
        mock_difficulty.calibrate.side_effect = difficulty_side_effect
    elif difficulty_return is not None:
        mock_difficulty.calibrate.return_value = difficulty_return
    else:
        mock_difficulty.calibrate.return_value = DifficultyCalibrationReport(
            aligned=True, target_theta=0.5, predicted_difficulty=0.4,
            delta=0.1, target_band="medium", predicted_band="medium",
            issues=[],
        )

    # Question repair mock — by default not repairable (no repair triggered)
    mock_question_repair = MagicMock()
    if question_repair_side_effect is not None:
        mock_question_repair.decide.side_effect = question_repair_side_effect
    elif question_repair_return is not None:
        mock_question_repair.decide.return_value = question_repair_return
    else:
        mock_question_repair.decide.return_value = RepairReport(
            repairable=False, attempts_remaining=0,
            failure_type="none", issues=[],
        )

    # Artifact validator mock — by default passes (citations + rationale valid)
    mock_artifact = MagicMock()
    if artifact_side_effect is not None:
        mock_artifact.validate.side_effect = artifact_side_effect
    elif artifact_return is not None:
        mock_artifact.validate.return_value = artifact_return
    else:
        mock_artifact.validate.return_value = ArtifactValidationReport(
            citations_valid=True, rationale_valid=True, issues=[],
        )

    patches = [
        patch("app.services.topic_streak_service.get_settings", return_value=settings),
    ]
    if langgraph:
        patches += [
            patch("app.services.adaptive_exam_service.get_settings", return_value=settings),
            patch("app.services.langgraph_rag_workflow.get_settings", return_value=settings),
            patch("app.services.langgraph_rag_workflow.RAGRetrievalService", return_value=mock_ret),
            patch("app.services.langgraph_rag_workflow.GeneratedQuestionService", return_value=mock_gen_svc),
            patch("app.services.langgraph_rag_workflow.GeneratedQuestionValidationService", return_value=mock_val_svc),
            patch("app.services.langgraph_rag_workflow.QueryPlannerService", return_value=mock_planner),
            patch("app.services.langgraph_rag_workflow.RerankerService", return_value=mock_reranker),
            patch("app.services.langgraph_rag_workflow.ContextCompressionService", return_value=mock_compressor),
            patch("app.services.langgraph_rag_workflow.RetrievalRepairService", return_value=mock_repair),
        ]
        if not use_real_dedup:
            patches.append(
                patch("app.services.langgraph_rag_workflow.QuestionDedupService", return_value=mock_dedup),
            )
        patches.append(
            patch("app.services.langgraph_rag_workflow.QuestionConfidenceService", return_value=mock_confidence),
        )
        patches.append(
            patch("app.services.langgraph_rag_workflow.GroundingValidationService", return_value=mock_grounding),
        )
        patches.append(
            patch("app.services.langgraph_rag_workflow.DistractorValidationService", return_value=mock_distractor),
        )
        patches.append(
            patch("app.services.langgraph_rag_workflow.DifficultyCalibrationService", return_value=mock_difficulty),
        )
        patches.append(
            patch("app.services.langgraph_rag_workflow.QuestionRepairService", return_value=mock_question_repair),
        )
        patches.append(
            patch("app.services.langgraph_rag_workflow.GeneratedArtifactValidationService", return_value=mock_artifact),
        )
    else:
        patches += [
            patch("app.services.adaptive_exam_service.get_settings", return_value=settings),
            patch("app.services.adaptive_exam_service.RAGRetrievalService", return_value=mock_ret),
            patch("app.services.adaptive_exam_service.GeneratedQuestionService", return_value=mock_gen_svc),
            patch("app.services.adaptive_exam_service.GeneratedQuestionValidationService", return_value=mock_val_svc),
        ]

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        result = _try_generate_next(
            db, ids["exam_id"], ids["topic_id"], 0.5, ids["student_id"],
        )
        if return_mocks:
            return result, mock_ret, mock_planner, mock_reranker, mock_gen_svc, mock_compressor, mock_repair, mock_dedup, mock_grounding, mock_distractor, mock_difficulty, mock_question_repair, mock_artifact
        return result


# ---------------------------------------------------------------------------
# Test results collector
# ---------------------------------------------------------------------------

_RESULTS: list[tuple[str, bool, str]] = []


def _ok(name: str) -> None:
    _RESULTS.append((name, True, ""))


def _fail(name: str, msg: str) -> None:
    _RESULTS.append((name, False, msg))


# ---------------------------------------------------------------------------
# 1. Legacy path — success
# ---------------------------------------------------------------------------

def test_legacy_success() -> None:
    NAME = "1. legacy_success"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])
        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=False),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
        )
        assert result is not None, "legacy path returned None"
        assert isinstance(result, GeneratedQuestion)
        assert result.topic_id == ids["topic_id"]
        assert result.source_exam_id == ids["exam_id"]
        assert result.text == "What is the primary purpose of encryption?"
        assert len(result.choices) == 4
        assert result.status.value == "auto_approved"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 2. LangGraph path — success
# ---------------------------------------------------------------------------

def test_langgraph_success() -> None:
    NAME = "2. langgraph_success"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])
        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
        )
        assert result is not None, "langgraph path returned None"
        assert isinstance(result, GeneratedQuestion)
        assert result.topic_id == ids["topic_id"]
        assert result.source_exam_id == ids["exam_id"]
        assert result.text == "What is the primary purpose of encryption?"
        assert result.status.value == "auto_approved"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 3. Streak gate blocks — both paths
# ---------------------------------------------------------------------------

def test_streak_gate_blocks() -> None:
    NAME = "3. streak_gate_blocks"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=2)  # below threshold of 4

        # Legacy
        r1 = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=False),
            chunks=_fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"]),
            gen=_fake_gen(),
            report=_valid_report(),
        )
        assert r1 is None, "legacy should return None when streak < threshold"

        # LangGraph
        r2 = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=_fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"]),
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
        )
        assert r2 is None, "langgraph should return None when streak < threshold"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 4. Empty retrieval — both paths
# ---------------------------------------------------------------------------

def test_empty_retrieval() -> None:
    NAME = "4. empty_retrieval"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)

        # Legacy
        r1 = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=False),
            chunks=[], gen=_fake_gen(), report=_valid_report(),
        )
        assert r1 is None, "legacy should return None on empty retrieval"

        # LangGraph
        r2 = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=[], gen=_fake_gen(), report=_valid_report(),
            langgraph=True,
        )
        assert r2 is None, "langgraph should return None on empty retrieval"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 5. Validation failure — no persistence, both paths
# ---------------------------------------------------------------------------

def test_validation_failure() -> None:
    NAME = "5. validation_failure"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])
        gen = _fake_gen()
        bad = _invalid_report()

        # Legacy
        r1 = _invoke(db, ids, _Settings(RAG_LANGGRAPH_ENABLED=False),
                      chunks=chunks, gen=gen, report=bad)
        assert r1 is None, "legacy should return None on validation failure"

        c1 = db.query(GeneratedQuestion).filter(
            GeneratedQuestion.source_exam_id == ids["exam_id"],
        ).count()
        assert c1 == 0, f"legacy: expected 0 persisted questions, got {c1}"

        # LangGraph
        r2 = _invoke(db, ids, _Settings(RAG_LANGGRAPH_ENABLED=True),
                      chunks=chunks, gen=gen, report=bad, langgraph=True)
        assert r2 is None, "langgraph should return None on validation failure"

        c2 = db.query(GeneratedQuestion).filter(
            GeneratedQuestion.source_exam_id == ids["exam_id"],
        ).count()
        assert c2 == 0, f"langgraph: expected 0 persisted questions, got {c2}"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 6. Success persists GeneratedQuestion + GeneratedQuestionEvidence
# ---------------------------------------------------------------------------

def test_persists_question_and_evidence() -> None:
    NAME = "6. persists_question_and_evidence"
    db = SessionLocal()
    try:
        # --- Legacy path ---
        ids_l = _setup(db)
        _set_streak(db, ids_l, streak=5)
        chunks_l = _fake_chunks(ids_l["topic_id"], n=3, chunk_ids=ids_l["chunk_ids"], document_id=ids_l["document_id"])

        result_l = _invoke(
            db, ids_l, _Settings(RAG_LANGGRAPH_ENABLED=False),
            chunks=chunks_l, gen=_fake_gen(), report=_valid_report(),
        )
        assert result_l is not None, "legacy: _try_generate_next returned None"
        qid_l = result_l.id

        q_l = db.query(GeneratedQuestion).filter(GeneratedQuestion.id == qid_l).first()
        assert q_l is not None, "legacy: GeneratedQuestion not in DB"
        assert q_l.topic_id == ids_l["topic_id"]
        assert q_l.source_exam_id == ids_l["exam_id"]
        assert q_l.text == "What is the primary purpose of encryption?"
        assert q_l.explanation == "Encryption converts plaintext to ciphertext."
        assert q_l.difficulty_estimate == 0.4
        assert q_l.validation_report is not None
        assert q_l.validation_report["schema_ok"] is True

        ev_l = db.query(GeneratedQuestionEvidence).filter(
            GeneratedQuestionEvidence.generated_question_id == qid_l,
        ).all()
        assert len(ev_l) == 3, f"legacy: expected 3 evidence rows, got {len(ev_l)}"
        chunk_id_set = {c.chunk_id for c in chunks_l}
        for e in ev_l:
            assert e.chunk_id in chunk_id_set, f"unexpected chunk_id {e.chunk_id}"

        # --- LangGraph path (fresh setup — streak gate resets) ---
        ids_g = _setup(db)
        _set_streak(db, ids_g, streak=5)
        chunks_g = _fake_chunks(ids_g["topic_id"], n=3, chunk_ids=ids_g["chunk_ids"], document_id=ids_g["document_id"])

        result_g = _invoke(
            db, ids_g, _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks_g, gen=_fake_gen(), report=_valid_report(),
            langgraph=True,
        )
        assert result_g is not None, "langgraph: _try_generate_next returned None"
        qid_g = result_g.id

        q_g = db.query(GeneratedQuestion).filter(GeneratedQuestion.id == qid_g).first()
        assert q_g is not None, "langgraph: GeneratedQuestion not in DB"
        assert q_g.topic_id == ids_g["topic_id"]
        assert q_g.source_exam_id == ids_g["exam_id"]

        ev_g = db.query(GeneratedQuestionEvidence).filter(
            GeneratedQuestionEvidence.generated_question_id == qid_g,
        ).all()
        assert len(ev_g) == 3, f"langgraph: expected 3 evidence rows, got {len(ev_g)}"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 7. Success increments generated_count
# ---------------------------------------------------------------------------

def test_increments_generated_count() -> None:
    NAME = "7. increments_generated_count"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5, generated=0)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        # --- Legacy path ---
        _invoke(
            db, ids, _Settings(RAG_LANGGRAPH_ENABLED=False),
            chunks=chunks, gen=_fake_gen(), report=_valid_report(),
        )
        p1 = db.query(StudentTopicProgress).filter(
            StudentTopicProgress.exam_id == ids["exam_id"],
            StudentTopicProgress.topic_id == ids["topic_id"],
        ).first()
        assert p1 is not None, "StudentTopicProgress not found (legacy)"
        assert p1.generated_count == 1, (
            f"legacy: expected generated_count=1, got {p1.generated_count}"
        )

        # --- LangGraph path (fresh data) ---
        db.rollback()
        ids2 = _setup(db)
        _set_streak(db, ids2, streak=5, generated=0)
        chunks2 = _fake_chunks(ids2["topic_id"], chunk_ids=ids2["chunk_ids"], document_id=ids2["document_id"])

        _invoke(
            db, ids2, _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks2, gen=_fake_gen(), report=_valid_report(),
            langgraph=True,
        )
        p2 = db.query(StudentTopicProgress).filter(
            StudentTopicProgress.exam_id == ids2["exam_id"],
            StudentTopicProgress.topic_id == ids2["topic_id"],
        ).first()
        assert p2 is not None, "StudentTopicProgress not found (langgraph)"
        assert p2.generated_count == 1, (
            f"langgraph: expected generated_count=1, got {p2.generated_count}"
        )

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 8. Equivalence — both paths produce identical DB artefacts
# ---------------------------------------------------------------------------

def test_equivalence() -> None:
    NAME = "8. equivalence"
    db = SessionLocal()
    try:
        gen = _fake_gen()
        report = _valid_report()

        # --- Run legacy ---
        ids_l = _setup(db)
        _set_streak(db, ids_l, streak=5)
        chunks_l = _fake_chunks(ids_l["topic_id"], chunk_ids=ids_l["chunk_ids"], document_id=ids_l["document_id"])
        res_l = _invoke(
            db, ids_l, _Settings(RAG_LANGGRAPH_ENABLED=False),
            chunks=chunks_l, gen=gen, report=report,
        )
        assert res_l is not None

        l_fields = {
            "text": res_l.text,
            "choices": res_l.choices,
            "explanation": res_l.explanation,
            "difficulty": res_l.difficulty_estimate,
            "status": res_l.status,
        }
        l_evidence_count = db.query(GeneratedQuestionEvidence).filter(
            GeneratedQuestionEvidence.generated_question_id == res_l.id,
        ).count()
        l_prog = db.query(StudentTopicProgress).filter(
            StudentTopicProgress.exam_id == ids_l["exam_id"],
            StudentTopicProgress.topic_id == ids_l["topic_id"],
        ).first()
        l_gen_count = l_prog.generated_count

        # --- Run LangGraph ---
        ids_g = _setup(db)
        _set_streak(db, ids_g, streak=5)
        chunks_g = _fake_chunks(ids_g["topic_id"], chunk_ids=ids_g["chunk_ids"], document_id=ids_g["document_id"])
        res_g = _invoke(
            db, ids_g, _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks_g, gen=gen, report=report, langgraph=True,
        )
        assert res_g is not None

        g_fields = {
            "text": res_g.text,
            "choices": res_g.choices,
            "explanation": res_g.explanation,
            "difficulty": res_g.difficulty_estimate,
            "status": res_g.status,
        }
        # Strip confidence keys added by the confidence gate (LangGraph only)
        # and langgraph_trace added by telemetry (LangGraph only)
        _LG_ONLY_KEYS = {"confidence_route_raw", "confidence_route_effective", "confidence_score", "langgraph_trace", "grounding", "distractor", "difficulty", "artifact", "evidence_citations", "distractor_rationale"}
        l_vr = {k: v for k, v in (res_l.validation_report or {}).items() if k not in _LG_ONLY_KEYS}
        g_vr = {k: v for k, v in (res_g.validation_report or {}).items() if k not in _LG_ONLY_KEYS}
        l_fields["validation_report"] = l_vr
        g_fields["validation_report"] = g_vr
        g_evidence_count = db.query(GeneratedQuestionEvidence).filter(
            GeneratedQuestionEvidence.generated_question_id == res_g.id,
        ).count()
        g_prog = db.query(StudentTopicProgress).filter(
            StudentTopicProgress.exam_id == ids_g["exam_id"],
            StudentTopicProgress.topic_id == ids_g["topic_id"],
        ).first()
        g_gen_count = g_prog.generated_count

        # --- Compare ---
        mismatches = []
        for key in l_fields:
            if l_fields[key] != g_fields[key]:
                mismatches.append(f"{key}: legacy={l_fields[key]!r} vs lg={g_fields[key]!r}")
        if l_evidence_count != g_evidence_count:
            mismatches.append(
                f"evidence_count: legacy={l_evidence_count} vs lg={g_evidence_count}"
            )
        if l_gen_count != g_gen_count:
            mismatches.append(
                f"generated_count: legacy={l_gen_count} vs lg={g_gen_count}"
            )

        assert not mismatches, "Differences found:\n" + "\n".join(mismatches)
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# =========================================================================
# Phase 2.1 — Query Planner Tests
# =========================================================================

# ---------------------------------------------------------------------------
# 9. Query planner creates multiple queries
# ---------------------------------------------------------------------------

def test_query_planner_creates_multiple_queries() -> None:
    NAME = "9. query_planner_creates_multiple_queries"
    try:
        planner = QueryPlannerService()
        plan = planner.plan(topic_name="Cryptography", theta=0.0)
        assert len(plan.queries) >= 2, f"expected >= 2 queries, got {len(plan.queries)}"
        assert plan.primary_query in plan.queries, "primary_query must appear in queries list"
        # All queries should be non-empty strings
        for q in plan.queries:
            assert isinstance(q, str) and q.strip(), f"empty query in list: {plan.queries!r}"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 10. Query planner is deterministic for same inputs
# ---------------------------------------------------------------------------

def test_query_planner_deterministic() -> None:
    NAME = "10. query_planner_deterministic"
    try:
        planner = QueryPlannerService()
        p1 = planner.plan(topic_name="Network Security", theta=0.7)
        p2 = planner.plan(topic_name="Network Security", theta=0.7)
        assert p1.queries == p2.queries, f"non-deterministic: {p1.queries} != {p2.queries}"
        assert p1.primary_query == p2.primary_query
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 11. Query planner difficulty bands shift primary query
# ---------------------------------------------------------------------------

def test_query_planner_difficulty_bands() -> None:
    NAME = "11. query_planner_difficulty_bands"
    try:
        planner = QueryPlannerService()
        easy = planner.plan(topic_name="Cryptography", theta=-1.0)
        med = planner.plan(topic_name="Cryptography", theta=0.0)
        hard = planner.plan(topic_name="Cryptography", theta=1.0)

        assert "basic fundamentals" in easy.primary_query.lower(), easy.primary_query
        assert easy.primary_query == med.primary_query or "advanced" not in easy.primary_query
        assert "advanced concepts" in hard.primary_query.lower() or "expert" in hard.primary_query.lower(), hard.primary_query
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 12. Multiple planned queries -> retrieve_multi called with those queries
# ---------------------------------------------------------------------------

def test_langgraph_multi_query_routing() -> None:
    NAME = "12. langgraph_multi_query_routing"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        planned = ["Advanced concepts in Cryptography", "common mistakes in Cryptography"]
        result, mock_ret, mock_planner, mock_reranker, mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            planned_queries=planned,
            return_mocks=True,
        )
        assert result is not None, "langgraph path returned None"

        # Planner was invoked
        mock_planner.plan.assert_called_once()

        # retrieve_multi was called (not retrieve)
        mock_ret.retrieve_multi.assert_called_once()
        mock_ret.retrieve.assert_not_called()

        # retrieve_multi received exactly the planned queries
        multi_args = mock_ret.retrieve_multi.call_args
        assert multi_args[0][0] == ids["topic_id"], "topic_id positional arg"
        assert multi_args[1]["queries"] == planned
        assert multi_args[1]["top_k"] == 5

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 13. Single planned query -> retrieve (not retrieve_multi) called
# ---------------------------------------------------------------------------

def test_langgraph_single_query_fallback() -> None:
    NAME = "13. langgraph_single_query_fallback"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result, mock_ret, _mock_planner, _mock_reranker, _mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            planned_queries=["Cryptography"],
            return_mocks=True,
        )
        assert result is not None, "single-query path returned None"

        # retrieve was called (not retrieve_multi)
        mock_ret.retrieve.assert_called_once()
        mock_ret.retrieve_multi.assert_not_called()

        # retrieve received the single planned query
        call_args = mock_ret.retrieve.call_args
        assert call_args[0][0] == ids["topic_id"], "topic_id positional arg"
        assert call_args[1]["query"] == "Cryptography"
        assert call_args[1]["top_k"] == 5

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 14. Empty planner output -> retrieve uses streak_info.topic_name
# ---------------------------------------------------------------------------

def test_langgraph_empty_planner_fallback() -> None:
    NAME = "14. langgraph_empty_planner_fallback"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result, mock_ret, _mock_planner, _mock_reranker, _mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            planned_queries=[],
            return_mocks=True,
        )
        assert result is not None, "empty-planner fallback returned None"

        # retrieve was called (not retrieve_multi)
        mock_ret.retrieve.assert_called_once()
        mock_ret.retrieve_multi.assert_not_called()

        # retrieve used streak_info.topic_name as query (not an empty string)
        call_args = mock_ret.retrieve.call_args
        query_used = call_args[1].get("query")
        assert query_used, f"expected a non-empty topic_name fallback, got {query_used!r}"
        assert call_args[0][0] == ids["topic_id"], "topic_id positional arg"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 15. Planner exception -> graph continues, falls back to single-query
# ---------------------------------------------------------------------------

def test_langgraph_planner_exception_fallback() -> None:
    NAME = "15. langgraph_planner_exception_fallback"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result, mock_ret, mock_planner, _mock_reranker, _mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            planner_side_effect=RuntimeError("planner exploded"),
            return_mocks=True,
        )
        assert result is not None, "planner exception should not kill the graph"

        # Planner was called but raised
        mock_planner.plan.assert_called_once()

        # Falls back to single-query retrieve (not retrieve_multi)
        mock_ret.retrieve.assert_called_once()
        mock_ret.retrieve_multi.assert_not_called()

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 16. Legacy path never enters the LangGraph branch
# ---------------------------------------------------------------------------

def test_legacy_path_never_uses_planner() -> None:
    NAME = "16. legacy_path_never_uses_planner"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        mock_run_rag_graph = MagicMock(return_value=None)

        # Patch run_rag_graph at its definition site — the lazy import
        # in _try_generate_next (line 278) resolves it from here.
        with patch(
            "app.services.langgraph_rag_workflow.run_rag_graph",
            mock_run_rag_graph,
        ):
            result, mock_ret, _mock_planner, _mock_reranker, _mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
                db, ids,
                _Settings(RAG_LANGGRAPH_ENABLED=False),
                chunks=chunks,
                gen=_fake_gen(),
                report=_valid_report(),
                langgraph=False,
                return_mocks=True,
            )

        assert result is not None, "legacy path returned None"

        # run_rag_graph is the sole entrypoint of the LangGraph branch.
        # If it was never called, the legacy branch was taken exclusively.
        mock_run_rag_graph.assert_not_called()

        # Legacy uses retrieve, not retrieve_multi
        mock_ret.retrieve.assert_called_once()
        mock_ret.retrieve_multi.assert_not_called()

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 17. Phase 1 equivalence still holds with planner in LangGraph path
# ---------------------------------------------------------------------------

def test_phase1_equivalence_with_planner() -> None:
    NAME = "17. phase1_equivalence_with_planner"
    db = SessionLocal()
    try:
        gen = _fake_gen()
        report = _valid_report()

        # --- Run legacy (no planner) ---
        ids_l = _setup(db)
        _set_streak(db, ids_l, streak=5)
        chunks_l = _fake_chunks(ids_l["topic_id"], chunk_ids=ids_l["chunk_ids"], document_id=ids_l["document_id"])
        res_l = _invoke(
            db, ids_l, _Settings(RAG_LANGGRAPH_ENABLED=False),
            chunks=chunks_l, gen=gen, report=report,
        )
        assert res_l is not None

        l_fields = {
            "text": res_l.text,
            "choices": res_l.choices,
            "explanation": res_l.explanation,
            "difficulty": res_l.difficulty_estimate,
            "status": res_l.status,
        }

        # --- Run LangGraph (with planner) ---
        ids_g = _setup(db)
        _set_streak(db, ids_g, streak=5)
        chunks_g = _fake_chunks(ids_g["topic_id"], chunk_ids=ids_g["chunk_ids"], document_id=ids_g["document_id"])
        res_g = _invoke(
            db, ids_g, _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks_g, gen=gen, report=report, langgraph=True,
        )
        assert res_g is not None

        g_fields = {
            "text": res_g.text,
            "choices": res_g.choices,
            "explanation": res_g.explanation,
            "difficulty": res_g.difficulty_estimate,
            "status": res_g.status,
        }

        mismatches = [k for k in l_fields if l_fields[k] != g_fields[k]]
        assert not mismatches, f"Differences found: {mismatches}"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# =========================================================================
# Phase 2.2 — Reranker Tests
# =========================================================================

# ---------------------------------------------------------------------------
# 18. Reranker node is invoked in LangGraph path
# ---------------------------------------------------------------------------

def test_reranker_node_invoked_in_langgraph() -> None:
    NAME = "18. reranker_node_invoked_in_langgraph"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result, _mock_ret, _mock_planner, mock_reranker, _mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            return_mocks=True,
        )
        assert result is not None, "langgraph path returned None"

        # Reranker was called exactly once
        mock_reranker.rerank.assert_called_once()

        # Reranker received the retrieved chunks
        rerank_args = mock_reranker.rerank.call_args
        assert len(rerank_args[0][0]) == 3, "reranker should receive 3 chunks"
        assert len(rerank_args[0][1]) >= 1, "reranker should receive at least 1 query"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 19. Reranker receives topic_name + candidate queries
# ---------------------------------------------------------------------------

def test_reranker_receives_correct_queries() -> None:
    NAME = "19. reranker_receives_correct_queries"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        # Get the actual topic name from DB (matches what StreakInfo.topic_name will be)
        topic = db.query(Topic).filter(Topic.id == ids["topic_id"]).first()
        topic_name = topic.name

        planned = ["Advanced Cryptography", "common mistakes in Cryptography"]
        result, _mock_ret, _mock_planner, mock_reranker, _mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            planned_queries=planned,
            return_mocks=True,
        )
        assert result is not None

        rerank_args = mock_reranker.rerank.call_args
        queries_received = rerank_args[0][1]

        # First query is streak_info.topic_name, then the planned queries
        assert queries_received[0] == topic_name, f"first query should be topic_name, got {queries_received[0]!r}"
        assert "Advanced Cryptography" in queries_received
        assert "common mistakes in Cryptography" in queries_received

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 20. Reranker exception falls back to original chunk order
# ---------------------------------------------------------------------------

def test_reranker_exception_fallback() -> None:
    NAME = "20. reranker_exception_fallback"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result, _mock_ret, _mock_planner, _mock_reranker, mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            reranker_side_effect=RuntimeError("reranker exploded"),
            return_mocks=True,
        )
        # Graph should still succeed — reranker is best-effort
        assert result is not None, "reranker exception should not kill the graph"
        assert result.topic_id == ids["topic_id"]

        # The original chunk order was passed into generate (not reranked)
        gen_input = mock_gen_svc.generate.call_args[0][0]
        received_ids = [c.chunk_id for c in gen_input.retrieved_chunks]
        original_ids = [c.chunk_id for c in chunks]
        assert received_ids == original_ids, (
            f"fallback should pass original order to generate: "
            f"got {received_ids}, expected {original_ids}"
        )

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 21. Lexical overlap scoring is deterministic
# ---------------------------------------------------------------------------

def test_lexical_overlap_deterministic() -> None:
    NAME = "21. lexical_overlap_deterministic"
    try:
        q = "encryption protects data confidentiality"
        c1 = "Encryption converts plaintext to ciphertext for confidentiality."
        c2 = "Compression reduces file size significantly."

        score1_a = lexical_overlap(q, c1)
        score1_b = lexical_overlap(q, c1)
        score2_a = lexical_overlap(q, c2)
        score2_b = lexical_overlap(q, c2)

        assert score1_a == score1_b, "lexical_overlap must be deterministic"
        assert score2_a == score2_b, "lexical_overlap must be deterministic"
        assert score1_a > score2_a, "chunk with more overlap should score higher"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 22. Legacy path never invokes reranker
# ---------------------------------------------------------------------------

def test_legacy_path_bypasses_reranker() -> None:
    NAME = "22. legacy_path_bypasses_reranker"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        mock_run_rag_graph = MagicMock(return_value=None)

        with patch(
            "app.services.langgraph_rag_workflow.run_rag_graph",
            mock_run_rag_graph,
        ):
            result, mock_ret, _mock_planner, mock_reranker, _mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
                db, ids,
                _Settings(RAG_LANGGRAPH_ENABLED=False),
                chunks=chunks,
                gen=_fake_gen(),
                report=_valid_report(),
                langgraph=False,
                return_mocks=True,
            )

        assert result is not None, "legacy path returned None"

        # run_rag_graph (the LangGraph entrypoint) was never called
        mock_run_rag_graph.assert_not_called()

        # Reranker was never invoked in the legacy path
        mock_reranker.rerank.assert_not_called()

        # Legacy uses retrieve directly
        mock_ret.retrieve.assert_called_once()

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 23. Reranked chunk order is what generate actually receives
# ---------------------------------------------------------------------------

def test_reranked_order_passed_to_generate() -> None:
    NAME = "23. reranked_order_passed_to_generate"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        # Reverse chunks to simulate a reranker that reorders
        reversed_chunks = list(reversed(chunks))

        # Build the state the reranker node would see after retrieve
        state = {
            "retrieved_chunks": list(chunks),
            "candidate_queries": ["Cryptography"],
            "streak_info": MagicMock(topic_name="Cryptography"),
        }

        # Patch RerankerService.rerank to return reversed order
        from app.services.langgraph_rag_workflow import reranker as reranker_node

        mock_svc = MagicMock()
        mock_svc.rerank.return_value = reversed_chunks
        with patch(
            "app.services.langgraph_rag_workflow.RerankerService",
            return_value=mock_svc,
        ):
            output = reranker_node(state)

        # The node must write the reranker output into retrieved_chunks
        result_ids = [c.chunk_id for c in output["retrieved_chunks"]]
        expected_ids = [c.chunk_id for c in reversed_chunks]
        assert result_ids == expected_ids, (
            f"reranker node should write reranked order to state: "
            f"got {result_ids}, expected {expected_ids}"
        )

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 24. Scoring-sensitive sort direction — highest score first
# ---------------------------------------------------------------------------

def test_reranker_sorts_highest_score_first() -> None:
    NAME = "24. reranker_sorts_highest_score_first"
    try:
        from app.services.retrieval_rerank_service import RerankerService

        # Three chunks with explicit similarity scores:
        #   chunk A: sim=0.9  (should rank 1st)
        #   chunk B: sim=0.3  (should rank 3rd)
        #   chunk C: sim=0.7  (should rank 2nd)
        chunks = [
            RetrievedChunk(
                chunk_id=1, document_id=0, course_name="test", title="A",
                text="encryption key management best practices",
                chunk_index=0, topic_id=1, similarity=0.9,
            ),
            RetrievedChunk(
                chunk_id=2, document_id=0, course_name="test", title="B",
                text="unrelated topic about gardening",
                chunk_index=1, topic_id=1, similarity=0.3,
            ),
            RetrievedChunk(
                chunk_id=3, document_id=0, course_name="test", title="C",
                text="encryption protects data in transit",
                chunk_index=2, topic_id=1, similarity=0.7,
            ),
        ]

        service = RerankerService()
        reranked = service.rerank(chunks, ["encryption"])

        assert len(reranked) == 3
        # Highest similarity should be first; lowest last
        assert reranked[0].chunk_id == 1, f"expected chunk_id=1 first, got {reranked[0].chunk_id}"
        assert reranked[1].chunk_id == 3, f"expected chunk_id=3 second, got {reranked[1].chunk_id}"
        assert reranked[2].chunk_id == 2, f"expected chunk_id=2 third, got {reranked[2].chunk_id}"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 25. Ties broken by original position (stable sort)
# ---------------------------------------------------------------------------

def test_reranker_tie_breaks_by_original_position() -> None:
    NAME = "25. reranker_tie_breaks_by_original_position"
    try:
        from app.services.retrieval_rerank_service import RerankerService

        # Two chunks with identical similarity and text — original order preserved
        chunks = [
            RetrievedChunk(
                chunk_id=10, document_id=0, course_name="test", title="First",
                text="encryption basics",
                chunk_index=0, topic_id=1, similarity=0.8,
            ),
            RetrievedChunk(
                chunk_id=20, document_id=0, course_name="test", title="Second",
                text="encryption basics",
                chunk_index=1, topic_id=1, similarity=0.8,
            ),
        ]

        service = RerankerService()
        reranked = service.rerank(chunks, ["encryption basics"])

        assert len(reranked) == 2
        assert reranked[0].chunk_id == 10, "earlier chunk should come first on tie"
        assert reranked[1].chunk_id == 20, "later chunk should come second on tie"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# =========================================================================
# Phase 2.3 — Context Compression + Evidence Gate Tests
# =========================================================================

# ---------------------------------------------------------------------------
# 26. Context compressor node is invoked in LangGraph path
# ---------------------------------------------------------------------------

def test_compressor_node_invoked_in_langgraph() -> None:
    NAME = "26. compressor_node_invoked_in_langgraph"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result, _mock_ret, _mock_planner, _mock_reranker, _mock_gen_svc, mock_comp, mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            return_mocks=True,
        )
        assert result is not None, "langgraph path returned None"

        # Compressor was called exactly once
        mock_comp.compress.assert_called_once()

        # Compressor received the reranker output (3 chunks)
        comp_args = mock_comp.compress.call_args
        assert len(comp_args[0][0]) == 3, "compressor should receive 3 chunks"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 27. Compressed chunks are what generation receives
# ---------------------------------------------------------------------------

def test_compressed_chunks_passed_to_generate() -> None:
    NAME = "27. compressed_chunks_passed_to_generate"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        # Compressor returns only the first chunk
        kept = [chunks[0]]

        result, _mock_ret, _mock_planner, _mock_reranker, mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            compressor_side_effect=lambda c, **kw: list(kept),
            return_mocks=True,
        )

        assert result is not None, "graph returned None"

        # generate() must receive the compressed output
        gen_input = mock_gen_svc.generate.call_args[0][0]
        received_ids = [c.chunk_id for c in gen_input.retrieved_chunks]
        expected_ids = [c.chunk_id for c in kept]
        assert received_ids == expected_ids, (
            f"generate should receive compressed chunks: "
            f"got {received_ids}, expected {expected_ids}"
        )

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 28. Evidence gate blocks generation when evidence is insufficient
# ---------------------------------------------------------------------------

def test_evidence_gate_blocks_generation() -> None:
    NAME = "28. evidence_gate_blocks_generation"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        # Force compressor to return empty list -> evidence gate blocks
        result, _mock_ret, _mock_planner, _mock_reranker, mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            compressor_side_effect=lambda c, **kw: [],
            return_mocks=True,
        )

        # Graph should exit without generating
        assert result is None, "evidence gate should block generation"

        # generate() was never called
        mock_gen_svc.generate.assert_not_called()

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 29. Safe early exit does not persist a generated question
# ---------------------------------------------------------------------------

def test_evidence_gate_no_persistence() -> None:
    NAME = "29. evidence_gate_no_persistence"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        # Force compressor to return empty list
        _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            compressor_side_effect=lambda c, **kw: [],
        )

        # No GeneratedQuestion should exist for this exam
        count = db.query(GeneratedQuestion).filter(
            GeneratedQuestion.source_exam_id == ids["exam_id"],
        ).count()
        assert count == 0, f"expected 0 persisted questions, got {count}"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 30. ContextCompressor exception falls back to original chunks
# ---------------------------------------------------------------------------

def test_compressor_exception_fallback() -> None:
    NAME = "30. compressor_exception_fallback"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result, _mock_ret, _mock_planner, _mock_reranker, mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            compressor_side_effect=RuntimeError("compressor exploded"),
            return_mocks=True,
        )

        # Graph should still succeed — compressor fallback preserves chunks
        assert result is not None, "compressor exception should not kill the graph"
        assert result.topic_id == ids["topic_id"]

        # generate() was called with the original chunks (fallback)
        gen_input = mock_gen_svc.generate.call_args[0][0]
        received_ids = [c.chunk_id for c in gen_input.retrieved_chunks]
        original_ids = [c.chunk_id for c in chunks]
        assert received_ids == original_ids, (
            f"fallback should pass original chunks to generate: "
            f"got {received_ids}, expected {original_ids}"
        )

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 31. EvidenceGate exception blocks generation (safety fallback)
# ---------------------------------------------------------------------------

def test_evidence_gate_exception_blocks() -> None:
    NAME = "31. evidence_gate_exception_blocks"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        mock_comp_svc = MagicMock()
        mock_comp_svc.compress.return_value = chunks

        mock_ev_svc = MagicMock()
        mock_ev_svc.validate.side_effect = RuntimeError("evidence gate exploded")

        with patch(
            "app.services.langgraph_rag_workflow.ContextCompressionService",
            return_value=mock_comp_svc,
        ), patch(
            "app.services.langgraph_rag_workflow.EvidenceValidationService",
            return_value=mock_ev_svc,
        ):
            result, _mock_ret, _mock_planner, _mock_reranker, mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
                db, ids,
                _Settings(RAG_LANGGRAPH_ENABLED=True),
                chunks=chunks,
                gen=_fake_gen(),
                report=_valid_report(),
                langgraph=True,
                return_mocks=True,
            )

        # Evidence gate exception should block generation (safe default)
        assert result is None, "evidence gate exception should block generation"
        mock_gen_svc.generate.assert_not_called()

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 32. Legacy path bypasses compressor and evidence gate
# ---------------------------------------------------------------------------

def test_legacy_path_bypasses_compressor_and_gate() -> None:
    NAME = "32. legacy_path_bypasses_compressor_and_gate"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        mock_run_rag_graph = MagicMock(return_value=None)

        with patch(
            "app.services.langgraph_rag_workflow.run_rag_graph",
            mock_run_rag_graph,
        ):
            result, mock_ret, _mock_planner, mock_reranker, _mock_gen_svc, mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
                db, ids,
                _Settings(RAG_LANGGRAPH_ENABLED=False),
                chunks=chunks,
                gen=_fake_gen(),
                report=_valid_report(),
                langgraph=False,
                return_mocks=True,
            )

        assert result is not None, "legacy path returned None"

        # LangGraph entrypoint was never called
        mock_run_rag_graph.assert_not_called()

        # Compressor was never invoked in the legacy path
        mock_comp.compress.assert_not_called()

        # Legacy uses retrieve directly
        mock_ret.retrieve.assert_called_once()

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 33. Context compression deduplicates near-identical chunks
# ---------------------------------------------------------------------------

def test_compression_deduplication() -> None:
    NAME = "33. compression_deduplication"
    try:
        topic_id = 1
        chunks = [
            RetrievedChunk(
                chunk_id=1, document_id=0, course_name="test", title="A",
                text="Encryption protects data confidentiality in transit and at rest for always",
                chunk_index=0, topic_id=topic_id, similarity=0.9,
            ),
            RetrievedChunk(
                chunk_id=2, document_id=0, course_name="test", title="B",
                text="Encryption protects data confidentiality in transit and at rest for always here",
                chunk_index=1, topic_id=topic_id, similarity=0.85,
            ),
            RetrievedChunk(
                chunk_id=3, document_id=0, course_name="test", title="C",
                text="Completely unrelated text about gardening and plants",
                chunk_index=2, topic_id=topic_id, similarity=0.3,
            ),
        ]

        service = ContextCompressionService()
        compressed = service.compress(chunks)

        # Chunks 1 and 2 are near-duplicates; chunk 3 is unrelated
        # After dedup: keep chunk 1 (higher sim) + chunk 3
        chunk_ids = [c.chunk_id for c in compressed]
        assert 1 in chunk_ids, "higher-sim duplicate should be kept"
        assert 3 in chunk_ids, "non-duplicate should be kept"
        assert 2 not in chunk_ids, "lower-sim duplicate should be removed"
        assert len(compressed) == 2, f"expected 2 chunks after dedup, got {len(compressed)}"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 34. Evidence validation unit — insufficient chunks
# ---------------------------------------------------------------------------

def test_evidence_validation_insufficient_chunks() -> None:
    NAME = "34. evidence_validation_insufficient_chunks"
    try:
        service = EvidenceValidationService()
        report = service.validate([])
        assert report.sufficient is False
        assert "Too few chunks" in report.reason

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 35. Evidence validation unit — low average similarity
# ---------------------------------------------------------------------------

def test_evidence_validation_low_similarity() -> None:
    NAME = "35. evidence_validation_low_similarity"
    try:
        chunks = [
            RetrievedChunk(
                chunk_id=1, document_id=0, course_name="test", title="A",
                text="tangent topic",
                chunk_index=0, topic_id=1, similarity=0.05,
            ),
            RetrievedChunk(
                chunk_id=2, document_id=0, course_name="test", title="B",
                text="another tangent",
                chunk_index=1, topic_id=1, similarity=0.08,
            ),
        ]

        service = EvidenceValidationService()
        report = service.validate(chunks)
        assert report.sufficient is False
        assert "Average similarity too low" in report.reason

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 36. Evidence validation unit — no high-quality chunks
# ---------------------------------------------------------------------------

def test_evidence_validation_no_high_quality() -> None:
    NAME = "36. evidence_validation_no_high_quality"
    try:
        chunks = [
            RetrievedChunk(
                chunk_id=1, document_id=0, course_name="test", title="A",
                text="somewhat related",
                chunk_index=0, topic_id=1, similarity=0.3,
            ),
            RetrievedChunk(
                chunk_id=2, document_id=0, course_name="test", title="B",
                text="also somewhat related",
                chunk_index=1, topic_id=1, similarity=0.25,
            ),
        ]

        service = EvidenceValidationService()
        report = service.validate(chunks)
        assert report.sufficient is False
        assert "No high-quality chunks" in report.reason

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 37. Evidence validation unit — sufficient
# ---------------------------------------------------------------------------

def test_evidence_validation_sufficient() -> None:
    NAME = "37. evidence_validation_sufficient"
    try:
        chunks = [
            RetrievedChunk(
                chunk_id=1, document_id=0, course_name="test", title="A",
                text="encryption key management best practices",
                chunk_index=0, topic_id=1, similarity=0.9,
            ),
            RetrievedChunk(
                chunk_id=2, document_id=0, course_name="test", title="B",
                text="encryption protects data in transit",
                chunk_index=1, topic_id=1, similarity=0.7,
            ),
        ]

        service = EvidenceValidationService()
        report = service.validate(chunks)
        assert report.sufficient is True
        assert report.chunk_count == 2
        assert report.high_quality_count == 2

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# =========================================================================
# Phase 2.4 — Retrieval Repair Tests
# =========================================================================

# ---------------------------------------------------------------------------
# 38. RetrievalRepairService generates broader queries
# ---------------------------------------------------------------------------

def test_retrieval_repair_service_generates_broader_queries() -> None:
    NAME = "38. retrieval_repair_service_generates_broader_queries"
    try:
        service = RetrievalRepairService()
        queries = service.repair(
            topic_name="Cryptography",
            original_queries=["advanced concepts in Cryptography"],
        )
        assert "Cryptography" in queries, "bare topic name should be included"
        assert any("basic fundamentals" in q for q in queries), "should include basic fundamentals"
        assert any("key principles" in q for q in queries), "should include key principles"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 39. RetrievalRepairService deduplicates against original queries
# ---------------------------------------------------------------------------

def test_retrieval_repair_service_deduplicates_against_original() -> None:
    NAME = "39. retrieval_repair_service_deduplicates_against_original"
    try:
        service = RetrievalRepairService()
        original = [
            "Cryptography",
            "key principles of Cryptography",
            "basic fundamentals of Cryptography",
        ]
        queries = service.repair(
            topic_name="Cryptography",
            original_queries=original,
        )
        # All three candidates are already in original, so only bare topic fallback
        assert queries == ["Cryptography"], f"expected bare topic fallback, got {queries}"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 40. RetrievalRepairService fallback when all candidates tried
# ---------------------------------------------------------------------------

def test_retrieval_repair_service_fallback_when_all_tried() -> None:
    NAME = "40. retrieval_repair_service_fallback_when_all_tried"
    try:
        service = RetrievalRepairService()
        queries = service.repair(
            topic_name="Cryptography",
            original_queries=[
                "Cryptography",
                "basic fundamentals of Cryptography",
                "key principles of Cryptography",
            ],
        )
        assert len(queries) == 1, f"expected exactly 1 fallback query, got {len(queries)}"
        assert queries[0] == "Cryptography", f"fallback should be bare topic, got {queries[0]}"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 41. Retrieval repair triggered after insufficient evidence
# ---------------------------------------------------------------------------

def test_retrieval_repair_triggered_after_insufficient_evidence() -> None:
    NAME = "41. retrieval_repair_triggered_after_insufficient_evidence"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        # Compressor returns [] on first call (evidence gate blocks -> repair),
        # then returns good chunks on second call (evidence gate passes -> generate)
        good_chunks = chunks
        call_count = 0

        def compressor_fn(c, **kw):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return []
            return list(good_chunks)

        result, _mock_ret, _mock_planner, _mock_reranker, _mock_gen_svc, _mock_comp, mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            compressor_side_effect=compressor_fn,
            return_mocks=True,
        )
        assert result is not None, "graph should succeed after repair"
        assert isinstance(result, GeneratedQuestion)
        mock_repair.repair.assert_called_once()
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 42. Success after repair persists question + evidence + increments count
# ---------------------------------------------------------------------------

def test_success_after_repair_persists() -> None:
    NAME = "42. success_after_repair_persists"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5, generated=0)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        good_chunks = chunks
        call_count = 0

        def compressor_fn(c, **kw):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return []
            return list(good_chunks)

        result, _mock_ret, _mock_planner, _mock_reranker, _mock_gen_svc, _mock_comp, mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            compressor_side_effect=compressor_fn,
            return_mocks=True,
        )

        assert result is not None, "graph should succeed after repair"
        mock_repair.repair.assert_called_once()
        qid = result.id

        # Verify persisted question
        q = db.query(GeneratedQuestion).filter(GeneratedQuestion.id == qid).first()
        assert q is not None, "GeneratedQuestion not in DB"
        assert q.topic_id == ids["topic_id"]
        assert q.source_exam_id == ids["exam_id"]

        # Verify evidence rows
        ev = db.query(GeneratedQuestionEvidence).filter(
            GeneratedQuestionEvidence.generated_question_id == qid,
        ).all()
        assert len(ev) == 3, f"expected 3 evidence rows, got {len(ev)}"

        # Verify generated_count incremented
        prog = db.query(StudentTopicProgress).filter(
            StudentTopicProgress.exam_id == ids["exam_id"],
            StudentTopicProgress.topic_id == ids["topic_id"],
        ).first()
        assert prog is not None, "StudentTopicProgress not found"
        assert prog.generated_count == 1, f"expected generated_count=1, got {prog.generated_count}"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 43. Double failure aborts safely — no persistence
# ---------------------------------------------------------------------------

def test_double_failure_aborts_safely() -> None:
    NAME = "43. double_failure_aborts_safely"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        # Compressor always returns [] — both evidence gates block
        result, _mock_ret, _mock_planner, _mock_reranker, mock_gen_svc, _mock_comp, mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            compressor_side_effect=lambda c, **kw: [],
            return_mocks=True,
        )

        assert result is None, "graph should return None after double failure"
        mock_repair.repair.assert_called_once()
        mock_gen_svc.generate.assert_not_called()

        # No question persisted
        count = db.query(GeneratedQuestion).filter(
            GeneratedQuestion.source_exam_id == ids["exam_id"],
        ).count()
        assert count == 0, f"expected 0 persisted questions, got {count}"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 44. Legacy path unchanged by repair — no repair service called
# ---------------------------------------------------------------------------

def test_legacy_path_unchanged_by_repair() -> None:
    NAME = "44. legacy_path_unchanged_by_repair"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        mock_run_rag_graph = MagicMock(return_value=None)

        with patch(
            "app.services.langgraph_rag_workflow.run_rag_graph",
            mock_run_rag_graph,
        ):
            result, mock_ret, _mock_planner, _mock_reranker, _mock_gen_svc, mock_comp, mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
                db, ids,
                _Settings(RAG_LANGGRAPH_ENABLED=False),
                chunks=chunks,
                gen=_fake_gen(),
                report=_valid_report(),
                langgraph=False,
                return_mocks=True,
            )

        assert result is not None, "legacy path returned None"
        mock_run_rag_graph.assert_not_called()
        mock_comp.compress.assert_not_called()
        mock_repair.repair.assert_not_called()
        mock_ret.retrieve.assert_called_once()

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 45. Repair exception fallback — graph ends safely, no persistence
# ---------------------------------------------------------------------------

def test_repair_exception_fallback() -> None:
    NAME = "45. repair_exception_fallback"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        # Compressor returns [] (evidence gate blocks -> repair),
        # repair raises -> graph ends safely with failure_reason
        result, _mock_ret, _mock_planner, _mock_reranker, mock_gen_svc, _mock_comp, mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            compressor_side_effect=lambda c, **kw: [],
            repair_side_effect=RuntimeError("repair exploded"),
            return_mocks=True,
        )

        assert result is None, "repair exception should cause graph to return None"
        mock_repair.repair.assert_called_once()
        mock_gen_svc.generate.assert_not_called()

        # No question persisted
        count = db.query(GeneratedQuestion).filter(
            GeneratedQuestion.source_exam_id == ids["exam_id"],
        ).count()
        assert count == 0, f"expected 0 persisted questions, got {count}"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# =========================================================================
# Phase 2.5 — Duplicate Gate Tests
# =========================================================================

# ---------------------------------------------------------------------------
# 46. Duplicate gate is invoked in LangGraph path
# ---------------------------------------------------------------------------

def test_duplicate_gate_invoked_in_langgraph() -> None:
    NAME = "46. duplicate_gate_invoked_in_langgraph"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result, _mock_ret, _mock_planner, _mock_reranker, _mock_gen_svc, _mock_comp, _mock_repair, mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            return_mocks=True,
        )
        assert result is not None, "langgraph path returned None"

        # Dedup service was called exactly once
        mock_dedup.check.assert_called_once()

        # Called with correct topic_id and the generated question text
        call_args = mock_dedup.check.call_args
        assert call_args[1]["topic_id"] == ids["topic_id"]
        assert call_args[1]["question_text"] == "What is the primary purpose of encryption?"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 47. Duplicate detected — blocks persistence
# ---------------------------------------------------------------------------

def test_duplicate_blocks_persistence() -> None:
    NAME = "47. duplicate_blocks_persistence"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        # Force dedup to report a duplicate
        dup_report = DedupReport(
            is_duplicate=True, max_similarity=0.88,
            source="generated", compared_count=3, threshold=0.65,
        )

        result, _mock_ret, _mock_planner, _mock_reranker, mock_gen_svc, _mock_comp, _mock_repair, mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            dedup_return=dup_report,
            return_mocks=True,
        )

        assert result is None, "duplicate gate should block persistence"
        mock_gen_svc.generate.assert_called()  # generate ran, but persist was blocked

        count = db.query(GeneratedQuestion).filter(
            GeneratedQuestion.source_exam_id == ids["exam_id"],
        ).count()
        assert count == 0, f"expected 0 persisted questions, got {count}"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 48. Non-duplicate still persists
# ---------------------------------------------------------------------------

def test_non_duplicate_persists() -> None:
    NAME = "48. non_duplicate_persists"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5, generated=0)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        # Non-duplicate report (default mock)
        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
        )

        assert result is not None, "non-duplicate should persist"
        assert isinstance(result, GeneratedQuestion)
        assert result.text == "What is the primary purpose of encryption?"

        # Verify evidence rows
        ev = db.query(GeneratedQuestionEvidence).filter(
            GeneratedQuestionEvidence.generated_question_id == result.id,
        ).all()
        assert len(ev) == 3, f"expected 3 evidence rows, got {len(ev)}"

        # Verify generated_count incremented
        prog = db.query(StudentTopicProgress).filter(
            StudentTopicProgress.exam_id == ids["exam_id"],
            StudentTopicProgress.topic_id == ids["topic_id"],
        ).first()
        assert prog.generated_count == 1

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 49. Duplicate generated question (real DB query) — blocked
# ---------------------------------------------------------------------------

def test_duplicate_generated_question_blocked() -> None:
    NAME = "49. duplicate_generated_question_blocked"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5, generated=0)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        # Insert an existing GeneratedQuestion with identical text for this topic
        from app.models.enums import GeneratedQuestionStatus
        existing_gq = GeneratedQuestion(
            topic_id=ids["topic_id"],
            source_exam_id=ids["exam_id"],
            text="What is the primary purpose of encryption?",
            choices=[{"text": "A", "is_correct": True}],
            explanation="existing",
            difficulty_estimate=0.5,
            status=GeneratedQuestionStatus.auto_approved,
            review_required=False,
        )
        db.add(existing_gq)
        db.flush()

        # Run the graph with the real QuestionDedupService (no mock)
        # The real service will query the DB, find the existing row,
        # compute Jaccard similarity = 1.0 (identical text), and block.
        result, _mock_ret, _mock_planner, _mock_reranker, _mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            use_real_dedup=True,
            return_mocks=True,
        )

        # Duplicate gate blocked: graph returned None
        assert result is None, "identical question text should be blocked by real dedup"

        # No new GeneratedQuestion persisted (only the pre-existing one exists)
        count = db.query(GeneratedQuestion).filter(
            GeneratedQuestion.topic_id == ids["topic_id"],
        ).count()
        assert count == 1, f"expected only the pre-existing row, got {count}"
        # Confirm it is the one we inserted
        existing = db.query(GeneratedQuestion).filter(
            GeneratedQuestion.topic_id == ids["topic_id"],
        ).first()
        assert existing.id == existing_gq.id, "should be the pre-existing row, not a new one"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 50. No persistence when duplicate detected (full DB check)
# ---------------------------------------------------------------------------

def test_no_persist_on_duplicate() -> None:
    NAME = "50. no_persist_on_duplicate"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        dup_report = DedupReport(
            is_duplicate=True, max_similarity=0.92,
            source="bank", compared_count=5, threshold=0.70,
        )

        _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            dedup_return=dup_report,
        )

        # No GeneratedQuestion for this exam
        count = db.query(GeneratedQuestion).filter(
            GeneratedQuestion.source_exam_id == ids["exam_id"],
        ).count()
        assert count == 0, f"expected 0 persisted questions, got {count}"

        # No evidence rows
        from app.models.rag import GeneratedQuestionEvidence
        ev_count = db.query(GeneratedQuestionEvidence).filter(
            GeneratedQuestionEvidence.generated_question_id == GeneratedQuestion.id,
            GeneratedQuestion.source_exam_id == ids["exam_id"],
        ).count()
        assert ev_count == 0, f"expected 0 evidence rows, got {ev_count}"

        # generated_count not incremented
        prog = db.query(StudentTopicProgress).filter(
            StudentTopicProgress.exam_id == ids["exam_id"],
            StudentTopicProgress.topic_id == ids["topic_id"],
        ).first()
        assert prog.generated_count == 0, f"generated_count should be 0, got {prog.generated_count}"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 51. Legacy path unchanged by duplicate gate
# ---------------------------------------------------------------------------

def test_legacy_path_unchanged_by_duplicate_gate() -> None:
    NAME = "51. legacy_path_unchanged_by_duplicate_gate"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        mock_run_rag_graph = MagicMock(return_value=None)

        # Patch the real LangGraph entrypoint — the sole bridge between
        # the legacy path and the graph.  If it is never called, the
        # legacy branch ran exclusively and the duplicate gate was never
        # reached.
        mock_dedup_cls = MagicMock()

        with patch(
            "app.services.langgraph_rag_workflow.run_rag_graph",
            mock_run_rag_graph,
        ), patch(
            "app.services.langgraph_rag_workflow.QuestionDedupService",
            mock_dedup_cls,
        ):
            result, mock_ret, _mock_planner, _mock_reranker, _mock_gen_svc, mock_comp, mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
                db, ids,
                _Settings(RAG_LANGGRAPH_ENABLED=False),
                chunks=chunks,
                gen=_fake_gen(),
                report=_valid_report(),
                langgraph=False,
                return_mocks=True,
            )

        assert result is not None, "legacy path returned None"

        # 1. The LangGraph entrypoint was never called
        mock_run_rag_graph.assert_not_called()

        # 2. QuestionDedupService was never instantiated
        mock_dedup_cls.assert_not_called()

        # 3. Legacy still uses retrieve directly (not retrieve_multi)
        mock_ret.retrieve.assert_called_once()
        mock_ret.retrieve_multi.assert_not_called()

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# =========================================================================
# Phase 2.6 — Confidence-Routing Gate Tests
# =========================================================================

# ---------------------------------------------------------------------------
# 52. Confidence gate is invoked in LangGraph path
# ---------------------------------------------------------------------------

def test_confidence_gate_invoked_in_langgraph() -> None:
    NAME = "52. confidence_gate_invoked_in_langgraph"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5, generated=0)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True, RAG_REVIEW_REQUIRED=False),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
        )

        assert result is not None, "langgraph path returned None"

        # The confidence gate ran and wrote its output into validation_report
        vr = result.validation_report
        assert vr is not None, "validation_report should be set"
        assert "confidence_route_raw" in vr, "confidence_route_raw missing — gate did not run"
        assert "confidence_route_effective" in vr, "confidence_route_effective missing — gate did not run"
        assert "confidence_score" in vr, "confidence_score missing — gate did not run"
        assert vr["confidence_route_raw"] == "auto_approve"
        assert vr["confidence_route_effective"] == "auto_approve"
        assert vr["confidence_score"] == 95.0

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 53. High confidence -> auto-approve (no global review)
# ---------------------------------------------------------------------------

def test_high_confidence_auto_approve() -> None:
    NAME = "53. high_confidence_auto_approve"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5, generated=0)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True, RAG_REVIEW_REQUIRED=False),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
        )

        assert result is not None, "high-confidence should persist"
        assert result.status.value == "auto_approved"
        assert result.review_required is False
        assert result.validation_report["confidence_route_raw"] == "auto_approve"
        assert result.validation_report["confidence_route_effective"] == "auto_approve"
        assert result.validation_report["confidence_score"] == 95.0

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 54. Medium confidence -> human_review / draft
# ---------------------------------------------------------------------------

def test_medium_confidence_human_review() -> None:
    NAME = "54. medium_confidence_human_review"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5, generated=0)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True, RAG_REVIEW_REQUIRED=False),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            confidence_return=ConfidenceReport(
                route="human_review", score=55.0,
                reasons=["Low avg evidence similarity (0.280)"],
            ),
        )

        assert result is not None, "medium-confidence should still persist"
        assert result.status.value == "draft"
        assert result.review_required is True
        assert result.validation_report["confidence_route_raw"] == "human_review"
        assert result.validation_report["confidence_route_effective"] == "human_review"
        assert result.validation_report["confidence_score"] == 55.0

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 55. Low confidence -> reject, no persistence
# ---------------------------------------------------------------------------

def test_low_confidence_reject() -> None:
    NAME = "55. low_confidence_reject"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            confidence_return=ConfidenceReport(
                route="reject", score=25.0,
                reasons=["Multiple quality issues"],
            ),
        )

        assert result is None, "low-confidence should reject"

        count = db.query(GeneratedQuestion).filter(
            GeneratedQuestion.source_exam_id == ids["exam_id"],
        ).count()
        assert count == 0, f"expected 0 persisted questions, got {count}"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 56. Global RAG_REVIEW_REQUIRED overrides auto-approve -> draft
# ---------------------------------------------------------------------------

def test_global_review_overrides_auto_approve() -> None:
    NAME = "56. global_review_overrides_auto_approve"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5, generated=0)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True, RAG_REVIEW_REQUIRED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            confidence_return=ConfidenceReport(
                route="auto_approve", score=95.0, reasons=[],
            ),
        )

        assert result is not None, "global review + auto_approve should still persist as draft"
        assert result.status.value == "draft"
        assert result.review_required is True
        # Raw route from confidence service was auto_approve, but global
        # RAG_REVIEW_REQUIRED overrode it to human_review at persist time.
        assert result.validation_report["confidence_route_raw"] == "auto_approve"
        assert result.validation_report["confidence_route_effective"] == "human_review"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 57. Confidence scoring is deterministic
# ---------------------------------------------------------------------------

def test_confidence_scoring_deterministic() -> None:
    NAME = "57. confidence_scoring_deterministic"
    try:
        chunks = [
            RetrievedChunk(
                chunk_id=1, document_id=0, course_name="test", title="A",
                text="encryption key management best practices",
                chunk_index=0, topic_id=1, similarity=0.9,
            ),
            RetrievedChunk(
                chunk_id=2, document_id=0, course_name="test", title="B",
                text="encryption protects data in transit",
                chunk_index=1, topic_id=1, similarity=0.7,
            ),
        ]
        v_report = ValidationReport(
            valid=True, issues=[], schema_ok=True,
            single_correct=True, non_duplicate=True, max_similarity=0.25,
        )
        gen = _fake_gen()

        service = QuestionConfidenceService()
        r1 = service.evaluate(
            retrieved_chunks=chunks, validation_report=v_report,
            retry_count=0, gen_output=gen,
        )
        r2 = service.evaluate(
            retrieved_chunks=chunks, validation_report=v_report,
            retry_count=0, gen_output=gen,
        )

        assert r1.route == r2.route, "route must be deterministic"
        assert r1.score == r2.score, "score must be deterministic"
        assert r1.reasons == r2.reasons, "reasons must be deterministic"
        assert r1.route == "auto_approve", "good inputs should auto-approve"
        assert r1.score >= 70.0, f"expected score >= 70, got {r1.score}"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 58. Legacy path unchanged by confidence gate
# ---------------------------------------------------------------------------

def test_legacy_path_unchanged_by_confidence_gate() -> None:
    NAME = "58. legacy_path_unchanged_by_confidence_gate"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        mock_run_rag_graph = MagicMock(return_value=None)
        mock_conf_cls = MagicMock()

        with patch(
            "app.services.langgraph_rag_workflow.run_rag_graph",
            mock_run_rag_graph,
        ), patch(
            "app.services.langgraph_rag_workflow.QuestionConfidenceService",
            mock_conf_cls,
        ):
            result, mock_ret, _mock_planner, _mock_reranker, _mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
                db, ids,
                _Settings(RAG_LANGGRAPH_ENABLED=False),
                chunks=chunks,
                gen=_fake_gen(),
                report=_valid_report(),
                langgraph=False,
                return_mocks=True,
            )

        assert result is not None, "legacy path returned None"

        # LangGraph entrypoint was never called
        mock_run_rag_graph.assert_not_called()

        # QuestionConfidenceService was never instantiated
        mock_conf_cls.assert_not_called()

        # Legacy still uses retrieve directly
        mock_ret.retrieve.assert_called_once()
        mock_ret.retrieve_multi.assert_not_called()

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 59. Confidence service exception -> fallback to human_review, persists
# ---------------------------------------------------------------------------

def test_confidence_exception_fallback_persists_draft() -> None:
    NAME = "59. confidence_exception_fallback_persists_draft"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5, generated=0)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True, RAG_REVIEW_REQUIRED=False),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            confidence_side_effect=RuntimeError("confidence service exploded"),
        )

        # Graph persists safely — exception fallback routes to human_review
        assert result is not None, "confidence exception should not kill the graph"
        assert isinstance(result, GeneratedQuestion)
        assert result.status.value == "draft", "exception fallback should produce draft"
        assert result.review_required is True, "exception fallback should require review"

        # Fallback metadata is present and consistent
        vr = result.validation_report
        assert vr is not None
        assert vr["confidence_route_raw"] == "human_review", "fallback raw route should be human_review"
        assert vr["confidence_route_effective"] == "human_review", "fallback effective route should be human_review"
        assert vr["confidence_score"] == 0.0, "fallback score should be 0.0"

        # Evidence rows still persisted
        ev = db.query(GeneratedQuestionEvidence).filter(
            GeneratedQuestionEvidence.generated_question_id == result.id,
        ).all()
        assert len(ev) == 3, f"expected 3 evidence rows, got {len(ev)}"

        # generated_count incremented
        prog = db.query(StudentTopicProgress).filter(
            StudentTopicProgress.exam_id == ids["exam_id"],
            StudentTopicProgress.topic_id == ids["topic_id"],
        ).first()
        assert prog.generated_count == 1

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 60. LangGraph path persists langgraph_trace in validation_report
# ---------------------------------------------------------------------------

def test_langgraph_trace_in_validation_report() -> None:
    NAME = "60. langgraph_trace_in_validation_report"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5, generated=0)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True, RAG_REVIEW_REQUIRED=False),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
        )

        assert result is not None, "graph should persist"
        vr = result.validation_report
        assert vr is not None, "validation_report must be set"
        assert "langgraph_trace" in vr, "validation_report must contain langgraph_trace"

        trace = vr["langgraph_trace"]
        assert isinstance(trace, dict), "langgraph_trace must be a dict"

        # trace_id present and non-empty
        assert isinstance(trace.get("trace_id"), str), "trace_id must be str"
        assert len(trace["trace_id"]) == 32, f"trace_id must be 32 hex chars, got {len(trace.get('trace_id', ''))}"

        # retry_count present
        assert "retry_count" in trace, "trace must include retry_count"

        # evidence sub-dict present with expected keys
        ev = trace.get("evidence")
        assert ev is not None, "trace must include evidence"
        assert "chunk_count" in ev, "evidence must include chunk_count"
        assert "avg_similarity" in ev, "evidence must include avg_similarity"
        assert "high_quality_count" in ev, "evidence must include high_quality_count"
        assert ev["chunk_count"] == 3, f"expected 3 chunks, got {ev['chunk_count']}"

        # confidence sub-dict present
        cr = trace.get("confidence")
        assert cr is not None, "trace must include confidence"
        assert cr["route_raw"] == "auto_approve", "raw route should be auto_approve"
        assert cr["route_effective"] == "auto_approve", "effective route should be auto_approve"
        assert isinstance(cr["score"], (int, float)), "score must be numeric"

        # validation sub-dict present
        val = trace.get("validation")
        assert val is not None, "trace must include validation"
        assert "schema_ok" in val, "validation must include schema_ok"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 61. Telemetry exception does not prevent persistence
# ---------------------------------------------------------------------------

def test_telemetry_exception_does_not_block_persist() -> None:
    NAME = "61. telemetry_exception_does_not_block_persist"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5, generated=0)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        with patch(
            "app.services.rag_telemetry_service.build_langgraph_trace",
            side_effect=RuntimeError("telemetry boom"),
        ):
            result = _invoke(
                db, ids,
                _Settings(RAG_LANGGRAPH_ENABLED=True, RAG_REVIEW_REQUIRED=False),
                chunks=chunks,
                gen=_fake_gen(),
                report=_valid_report(),
                langgraph=True,
            )

        # Graph persists despite telemetry failure
        assert result is not None, "telemetry exception must not block persist"
        assert isinstance(result, GeneratedQuestion)
        assert result.status.value == "auto_approved", "should still auto-approve"

        # Core validation fields survive even when trace build fails
        vr = result.validation_report
        assert vr is not None
        assert vr["schema_ok"] is True, "core validation fields must survive"
        assert vr["confidence_route_raw"] == "auto_approve"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# =========================================================================
# Phase 2.8 — Grounding Validator Tests
# =========================================================================

# ---------------------------------------------------------------------------
# 62. Grounding validator unit — well-supported question
# ---------------------------------------------------------------------------

def test_grounding_well_supported() -> None:
    NAME = "62. grounding_well_supported"
    try:
        chunks = [
            RetrievedChunk(
                chunk_id=1, document_id=0, course_name="test", title="A",
                text="Encryption protects data confidentiality by converting plaintext to ciphertext.",
                chunk_index=0, topic_id=1, similarity=0.9,
            ),
            RetrievedChunk(
                chunk_id=2, document_id=0, course_name="test", title="B",
                text="The primary purpose of encryption is to protect confidentiality.",
                chunk_index=1, topic_id=1, similarity=0.85,
            ),
        ]
        service = GroundingValidationService()
        report = service.validate(
            question_text="What is the primary purpose of encryption?",
            correct_answer_text="Protect confidentiality",
            explanation="Encryption converts plaintext to ciphertext.",
            retrieved_chunks=chunks,
        )
        assert report.grounded is True, f"expected grounded=True, got {report.grounded}"
        assert report.question_supported is True
        assert report.answer_supported is True
        assert report.explanation_supported is True
        assert report.support_score >= 0.10, f"expected score >= 0.10, got {report.support_score}"
        assert len(report.issues) == 0, f"expected no issues, got {report.issues}"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 63. Grounding validator unit — unsupported correct answer
# ---------------------------------------------------------------------------

def test_grounding_unsupported_answer() -> None:
    NAME = "63. grounding_unsupported_answer"
    try:
        chunks = [
            RetrievedChunk(
                chunk_id=1, document_id=0, course_name="test", title="A",
                text="Photosynthesis converts sunlight into chemical energy in plants.",
                chunk_index=0, topic_id=1, similarity=0.9,
            ),
        ]
        service = GroundingValidationService()
        report = service.validate(
            question_text="What is the primary purpose of encryption?",
            correct_answer_text="Protect confidentiality through advanced cryptographic algorithms",
            explanation="Encryption converts plaintext to ciphertext.",
            retrieved_chunks=chunks,
        )
        # Answer is completely unsupported — no overlap with photosynthesis
        assert report.grounded is False, f"expected grounded=False for unsupported answer"
        assert report.answer_supported is False
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 64. Grounding validator unit — unsupported explanation
# ---------------------------------------------------------------------------

def test_grounding_unsupported_explanation() -> None:
    NAME = "64. grounding_unsupported_explanation"
    try:
        chunks = [
            RetrievedChunk(
                chunk_id=1, document_id=0, course_name="test", title="A",
                text="Encryption protects confidentiality of data.",
                chunk_index=0, topic_id=1, similarity=0.9,
            ),
        ]
        service = GroundingValidationService()
        report = service.validate(
            question_text="What is the primary purpose of encryption?",
            correct_answer_text="Protect confidentiality",
            explanation="Quantum entanglement allows instantaneous communication across galaxies.",
            retrieved_chunks=chunks,
        )
        # Explanation is about quantum physics, not encryption
        assert report.grounded is False, f"expected grounded=False for unsupported explanation"
        assert report.explanation_supported is False
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 65. Grounding validator unit — no chunks
# ---------------------------------------------------------------------------

def test_grounding_no_chunks() -> None:
    NAME = "65. grounding_no_chunks"
    try:
        service = GroundingValidationService()
        report = service.validate(
            question_text="What is encryption?",
            correct_answer_text="Protect confidentiality",
            explanation="Encryption converts plaintext.",
            retrieved_chunks=[],
        )
        assert report.grounded is False
        assert report.question_supported is False
        assert report.answer_supported is False
        assert report.explanation_supported is False
        assert "No evidence chunks" in report.issues[0]
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 66. Grounding validator is invoked in LangGraph path
# ---------------------------------------------------------------------------

def test_grounding_validator_invoked_in_langgraph() -> None:
    NAME = "66. grounding_validator_invoked_in_langgraph"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result, _mock_ret, _mock_planner, _mock_reranker, _mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            return_mocks=True,
        )
        assert result is not None, "langgraph path returned None"

        # Grounding service was called exactly once
        mock_grounding.validate.assert_called_once()

        # Called with correct arguments
        call_args = mock_grounding.validate.call_args
        assert call_args[1]["question_text"] == "What is the primary purpose of encryption?"
        assert call_args[1]["correct_answer_text"] == "Protect confidentiality"
        assert call_args[1]["explanation"] == "Encryption converts plaintext to ciphertext."

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 67. Grounding failure blocks persistence
# ---------------------------------------------------------------------------

def test_grounding_failure_blocks_persistence() -> None:
    NAME = "67. grounding_failure_blocks_persistence"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        # Force grounding to fail
        ungrouned = GroundingReport(
            grounded=False, question_supported=False,
            answer_supported=False, explanation_supported=False,
            support_score=0.02,
            issues=["Correct answer support below threshold"],
        )

        result, _mock_ret, _mock_planner, _mock_reranker, mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            grounding_return=ungrouned,
            return_mocks=True,
        )

        assert result is None, "grounding failure should block persistence"
        mock_gen_svc.generate.assert_called()  # generate ran, but grounding blocked

        count = db.query(GeneratedQuestion).filter(
            GeneratedQuestion.source_exam_id == ids["exam_id"],
        ).count()
        assert count == 0, f"expected 0 persisted questions, got {count}"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 68. Grounding exception does not persist — fail-closed
# ---------------------------------------------------------------------------

def test_grounding_exception_blocks_persist() -> None:
    NAME = "68. grounding_exception_blocks_persist"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            grounding_side_effect=RuntimeError("grounding service exploded"),
        )

        # Fail-closed: exception blocks persistence
        assert result is None, "grounding exception should block persistence"

        count = db.query(GeneratedQuestion).filter(
            GeneratedQuestion.source_exam_id == ids["exam_id"],
        ).count()
        assert count == 0, f"expected 0 persisted questions, got {count}"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 69. Grounding report appears in persisted validation_report
# ---------------------------------------------------------------------------

def test_grounding_report_in_validation_report() -> None:
    NAME = "69. grounding_report_in_validation_report"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5, generated=0)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True, RAG_REVIEW_REQUIRED=False),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
        )

        assert result is not None, "graph should persist"
        vr = result.validation_report
        assert vr is not None, "validation_report must be set"
        assert "grounding" in vr, "validation_report must contain grounding"

        grounding = vr["grounding"]
        assert grounding["grounded"] is True
        assert grounding["question_supported"] is True
        assert grounding["answer_supported"] is True
        assert grounding["explanation_supported"] is True
        assert isinstance(grounding["support_score"], (int, float))
        assert isinstance(grounding["issues"], list)

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 70. Legacy path unchanged by grounding validator
# ---------------------------------------------------------------------------

def test_legacy_path_unchanged_by_grounding() -> None:
    NAME = "70. legacy_path_unchanged_by_grounding"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        mock_run_rag_graph = MagicMock(return_value=None)
        mock_ground_cls = MagicMock()

        with patch(
            "app.services.langgraph_rag_workflow.run_rag_graph",
            mock_run_rag_graph,
        ), patch(
            "app.services.langgraph_rag_workflow.GroundingValidationService",
            mock_ground_cls,
        ):
            result, mock_ret, _mock_planner, _mock_reranker, _mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
                db, ids,
                _Settings(RAG_LANGGRAPH_ENABLED=False),
                chunks=chunks,
                gen=_fake_gen(),
                report=_valid_report(),
                langgraph=False,
                return_mocks=True,
            )

        assert result is not None, "legacy path returned None"

        # LangGraph entrypoint was never called
        mock_run_rag_graph.assert_not_called()

        # GroundingValidationService was never instantiated
        mock_ground_cls.assert_not_called()

        # Legacy still uses retrieve directly
        mock_ret.retrieve.assert_called_once()

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 71. Unsupported question stem blocks grounding (answer IS supported)
# ---------------------------------------------------------------------------

def test_grounding_unsupported_stem_blocks() -> None:
    NAME = "71. grounding_unsupported_stem_blocks"
    try:
        chunks = [
            RetrievedChunk(
                chunk_id=1, document_id=0, course_name="test", title="A",
                text="Protect confidentiality is the key goal of encryption methods.",
                chunk_index=0, topic_id=1, similarity=0.9,
            ),
        ]
        service = GroundingValidationService()
        report = service.validate(
            question_text="What is photosynthesis and how does it work?",
            correct_answer_text="Protect confidentiality",
            explanation="Encryption protects data confidentiality.",
            retrieved_chunks=chunks,
        )
        # Stem is about photosynthesis (not in evidence), answer IS supported
        assert report.grounded is False, "unsupported stem should block grounding"
        assert report.question_supported is False
        assert report.answer_supported is True
        assert report.explanation_supported is True
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 72. Empty explanation blocks grounding
# ---------------------------------------------------------------------------

def test_grounding_empty_explanation_blocks() -> None:
    NAME = "72. grounding_empty_explanation_blocks"
    try:
        chunks = [
            RetrievedChunk(
                chunk_id=1, document_id=0, course_name="test", title="A",
                text="Encryption protects data confidentiality by converting plaintext.",
                chunk_index=0, topic_id=1, similarity=0.9,
            ),
        ]
        service = GroundingValidationService()
        report = service.validate(
            question_text="What is the primary purpose of encryption?",
            correct_answer_text="Protect confidentiality",
            explanation="",
            retrieved_chunks=chunks,
        )
        assert report.grounded is False, "empty explanation should block grounding"
        assert report.explanation_supported is False
        assert any("no content words" in i.lower() for i in report.issues)
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 73. Empty/stopword-only correct answer blocks grounding
# ---------------------------------------------------------------------------

def test_grounding_empty_answer_blocks() -> None:
    NAME = "73. grounding_empty_answer_blocks"
    try:
        chunks = [
            RetrievedChunk(
                chunk_id=1, document_id=0, course_name="test", title="A",
                text="Encryption protects data confidentiality by converting plaintext.",
                chunk_index=0, topic_id=1, similarity=0.9,
            ),
        ]
        service = GroundingValidationService()
        # Empty string
        r1 = service.validate(
            question_text="What is encryption?",
            correct_answer_text="",
            explanation="Encryption converts plaintext to ciphertext.",
            retrieved_chunks=chunks,
        )
        assert r1.grounded is False, "empty answer should block grounding"
        assert r1.answer_supported is False

        # Stopword-only string (all words are stop words)
        r2 = service.validate(
            question_text="What is encryption?",
            correct_answer_text="the is a an",
            explanation="Encryption converts plaintext to ciphertext.",
            retrieved_chunks=chunks,
        )
        assert r2.grounded is False, "stopword-only answer should block grounding"
        assert r2.answer_supported is False
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 74. Empty question stem blocks grounding
# ---------------------------------------------------------------------------

def test_grounding_empty_question_stem_blocks() -> None:
    NAME = "74. grounding_empty_question_stem_blocks"
    try:
        chunks = [
            RetrievedChunk(
                chunk_id=1, document_id=0, course_name="test", title="A",
                text="Encryption protects data confidentiality by converting plaintext.",
                chunk_index=0, topic_id=1, similarity=0.9,
            ),
        ]
        service = GroundingValidationService()
        # Empty question text
        r1 = service.validate(
            question_text="",
            correct_answer_text="Protect confidentiality",
            explanation="Encryption converts plaintext to ciphertext.",
            retrieved_chunks=chunks,
        )
        assert r1.grounded is False, "empty question stem should block grounding"
        assert r1.question_supported is False

        # Stopword-only question
        r2 = service.validate(
            question_text="is the a an it",
            correct_answer_text="Protect confidentiality",
            explanation="Encryption converts plaintext to ciphertext.",
            retrieved_chunks=chunks,
        )
        assert r2.grounded is False, "stopword-only question stem should block grounding"
        assert r2.question_supported is False
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# =========================================================================
# Phase 2.9 — Distractor Validator Tests
# =========================================================================

# ---------------------------------------------------------------------------
# 75. Distractor validator unit — valid distractors pass
# ---------------------------------------------------------------------------

def test_distractor_valid_passes() -> None:
    NAME = "75. distractor_valid_passes"
    try:
        options = [
            {"text": "Protect confidentiality", "is_correct": True},
            {"text": "Increase speed", "is_correct": False},
            {"text": "Compress files", "is_correct": False},
            {"text": "Generate randomness", "is_correct": False},
        ]
        service = DistractorValidationService()
        report = service.validate(options)
        assert report.valid is True, f"expected valid=True, got {report.valid}"
        assert report.distinct_distractors is True
        assert report.separated_from_correct is True
        assert report.meaningful_distractors is True
        assert len(report.issues) == 0, f"expected no issues, got {report.issues}"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 76. Distractor validator unit — duplicate distractors block
# ---------------------------------------------------------------------------

def test_distractor_duplicates_block() -> None:
    NAME = "76. distractor_duplicates_block"
    try:
        options = [
            {"text": "Protect confidentiality", "is_correct": True},
            {"text": "Encrypt data to protect privacy of information", "is_correct": False},
            {"text": "Encrypt data to protect the privacy of information", "is_correct": False},
            {"text": "Compress files", "is_correct": False},
        ]
        service = DistractorValidationService()
        report = service.validate(options)
        assert report.valid is False, "near-duplicate distractors should fail"
        assert report.distinct_distractors is False
        assert any("near-duplicate" in i.lower() for i in report.issues)
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 77. Distractor validator unit — distractor too similar to correct answer
# ---------------------------------------------------------------------------

def test_distractor_too_similar_to_answer_blocks() -> None:
    NAME = "77. distractor_too_similar_to_answer_blocks"
    try:
        options = [
            {"text": "Protect confidentiality of data", "is_correct": True},
            {"text": "Protect data confidentiality", "is_correct": False},
            {"text": "Compress files", "is_correct": False},
            {"text": "Generate randomness", "is_correct": False},
        ]
        service = DistractorValidationService()
        report = service.validate(options)
        assert report.valid is False, "distractor too close to correct should fail"
        assert report.separated_from_correct is False
        assert any("too similar" in i.lower() for i in report.issues)
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 78. Distractor validator unit — empty distractor blocks
# ---------------------------------------------------------------------------

def test_distractor_empty_blocks() -> None:
    NAME = "78. distractor_empty_blocks"
    try:
        options = [
            {"text": "Protect confidentiality", "is_correct": True},
            {"text": "", "is_correct": False},
            {"text": "Compress files", "is_correct": False},
        ]
        service = DistractorValidationService()
        report = service.validate(options)
        assert report.valid is False, "empty distractor should fail"
        assert report.meaningful_distractors is False
        assert any("empty" in i.lower() for i in report.issues)
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 79. Distractor validator unit — too few distractors blocks
# ---------------------------------------------------------------------------

def test_distractor_too_few_blocks() -> None:
    NAME = "79. distractor_too_few_blocks"
    try:
        options = [
            {"text": "Protect confidentiality", "is_correct": True},
            {"text": "Increase speed", "is_correct": False},
        ]
        service = DistractorValidationService()
        report = service.validate(options)
        assert report.valid is False, "only 1 distractor should fail"
        assert any("too few" in i.lower() for i in report.issues)
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 80. Distractor validator is invoked in LangGraph path
# ---------------------------------------------------------------------------

def test_distractor_validator_invoked_in_langgraph() -> None:
    NAME = "80. distractor_validator_invoked_in_langgraph"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result, _mock_ret, _mock_planner, _mock_reranker, _mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            return_mocks=True,
        )
        assert result is not None, "langgraph path returned None"

        # Distractor service was called exactly once
        mock_distractor.validate.assert_called_once()

        # Called with the generated options
        call_args = mock_distractor.validate.call_args
        options_received = call_args[1]["options"]
        assert len(options_received) == 4, "should receive 4 options"
        assert options_received[0]["is_correct"] is True

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 81. Distractor failure blocks persistence — no question saved
# ---------------------------------------------------------------------------

def test_distractor_failure_blocks_persistence() -> None:
    NAME = "81. distractor_failure_blocks_persistence"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        bad_distractor_report = DistractorReport(
            valid=False, distinct_distractors=False,
            separated_from_correct=True, meaningful_distractors=True,
            issues=["Distractors 1 and 2 are near-duplicates (Jaccard=0.90)"],
        )

        result, _mock_ret, _mock_planner, _mock_reranker, mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            distractor_return=bad_distractor_report,
            return_mocks=True,
        )

        assert result is None, "distractor failure should block persistence"
        mock_gen_svc.generate.assert_called()  # generate ran, but distractor blocked

        count = db.query(GeneratedQuestion).filter(
            GeneratedQuestion.source_exam_id == ids["exam_id"],
        ).count()
        assert count == 0, f"expected 0 persisted questions, got {count}"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 82. Distractor report appears in persisted validation_report
# ---------------------------------------------------------------------------

def test_distractor_report_in_validation_report() -> None:
    NAME = "82. distractor_report_in_validation_report"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5, generated=0)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True, RAG_REVIEW_REQUIRED=False),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
        )

        assert result is not None, "graph should persist"
        vr = result.validation_report
        assert vr is not None, "validation_report must be set"
        assert "distractor" in vr, "validation_report must contain distractor"

        distractor = vr["distractor"]
        assert distractor["valid"] is True
        assert distractor["distinct_distractors"] is True
        assert distractor["separated_from_correct"] is True
        assert distractor["meaningful_distractors"] is True
        assert isinstance(distractor["issues"], list)

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 83. Legacy path unchanged by distractor validator
# ---------------------------------------------------------------------------

def test_legacy_path_unchanged_by_distractor() -> None:
    NAME = "83. legacy_path_unchanged_by_distractor"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        mock_run_rag_graph = MagicMock(return_value=None)
        mock_distr_cls = MagicMock()

        with patch(
            "app.services.langgraph_rag_workflow.run_rag_graph",
            mock_run_rag_graph,
        ), patch(
            "app.services.langgraph_rag_workflow.DistractorValidationService",
            mock_distr_cls,
        ):
            result, mock_ret, _mock_planner, _mock_reranker, _mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
                db, ids,
                _Settings(RAG_LANGGRAPH_ENABLED=False),
                chunks=chunks,
                gen=_fake_gen(),
                report=_valid_report(),
                langgraph=False,
                return_mocks=True,
            )

        assert result is not None, "legacy path returned None"

        # LangGraph entrypoint was never called
        mock_run_rag_graph.assert_not_called()

        # DistractorValidationService was never instantiated
        mock_distr_cls.assert_not_called()

        # Legacy still uses retrieve directly
        mock_ret.retrieve.assert_called_once()

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# =========================================================================
# Phase 2.9b — Distractor Validator Hardening Tests
# =========================================================================

# ---------------------------------------------------------------------------
# 84. Empty correct answer blocks distractor validation (unit)
# ---------------------------------------------------------------------------

def test_distractor_empty_correct_answer_blocks() -> None:
    NAME = "84. distractor_empty_correct_answer_blocks"
    try:
        options = [
            {"text": "", "is_correct": True},
            {"text": "Increase speed", "is_correct": False},
            {"text": "Compress files", "is_correct": False},
            {"text": "Generate randomness", "is_correct": False},
        ]
        service = DistractorValidationService()
        report = service.validate(options)
        assert report.valid is False, "empty correct answer should fail validation"
        assert report.separated_from_correct is False
        assert any("correct answer text is empty" in i.lower() for i in report.issues)
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 85. Stopword-only distractor fails meaningful check
# ---------------------------------------------------------------------------

def test_distractor_stopword_only_blocks() -> None:
    NAME = "85. distractor_stopword_only_blocks"
    try:
        options = [
            {"text": "Protect confidentiality", "is_correct": True},
            {"text": "the and of", "is_correct": False},
            {"text": "Compress files", "is_correct": False},
            {"text": "Generate randomness", "is_correct": False},
        ]
        service = DistractorValidationService()
        report = service.validate(options)
        assert report.valid is False, "stopword-only distractor should fail"
        assert report.meaningful_distractors is False
        assert any("no content words" in i.lower() for i in report.issues)
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 86. Empty correct answer blocks distractor validator node (direct invocation)
# ---------------------------------------------------------------------------

def test_distractor_empty_correct_answer_blocks_persist() -> None:
    NAME = "86. distractor_empty_correct_answer_blocks_persist"
    try:
        from app.services.langgraph_rag_workflow import distractor_validator

        gen_empty_answer = GenerationOutput(
            question_text="What is the primary purpose of encryption?",
            options=[
                {"text": "", "is_correct": True},
                {"text": "Increase speed", "is_correct": False},
                {"text": "Compress files", "is_correct": False},
                {"text": "Generate randomness", "is_correct": False},
            ],
            explanation="Encryption converts plaintext to ciphertext.",
            difficulty_estimate=0.4,
        )

        state = {"gen_output": gen_empty_answer}
        result = distractor_validator(state)

        assert "failure_reason" in result, "node should set failure_reason"
        assert result["distractor_report"].valid is False
        assert result["distractor_report"].separated_from_correct is False

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# =========================================================================
# Phase 2.10 — Difficulty Calibration Tests
# =========================================================================

# ---------------------------------------------------------------------------
# 87. Aligned difficulty passes through (unit)
# ---------------------------------------------------------------------------

def test_difficulty_aligned_passes() -> None:
    NAME = "87. difficulty_aligned_passes"
    try:
        service = DifficultyCalibrationService()
        report = service.calibrate(target_theta=0.5, difficulty_estimate=0.4)
        assert report.aligned is True, "aligned difficulty should pass"
        assert report.delta is not None
        assert report.delta < 0.2
        assert report.target_band == "medium"
        assert report.predicted_band == "medium"
        assert report.issues == []
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 88. Overly easy question blocks (unit)
# ---------------------------------------------------------------------------

def test_difficulty_too_easy_blocks() -> None:
    NAME = "88. difficulty_too_easy_blocks"
    try:
        service = DifficultyCalibrationService()
        report = service.calibrate(target_theta=1.0, difficulty_estimate=-1.5)
        assert report.aligned is False, "too-easy question should be blocked"
        assert report.delta is not None
        assert report.delta > 2.0
        assert report.target_band == "hard"
        assert report.predicted_band == "easy"
        assert any("delta" in i.lower() for i in report.issues)
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 89. Overly hard question blocks (unit)
# ---------------------------------------------------------------------------

def test_difficulty_too_hard_blocks() -> None:
    NAME = "89. difficulty_too_hard_blocks"
    try:
        service = DifficultyCalibrationService()
        report = service.calibrate(target_theta=-1.0, difficulty_estimate=1.5)
        assert report.aligned is False, "too-hard question should be blocked"
        assert report.delta is not None
        assert report.delta > 2.0
        assert report.target_band == "easy"
        assert report.predicted_band == "hard"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 90. Missing difficulty estimate blocks (unit)
# ---------------------------------------------------------------------------

def test_difficulty_missing_estimate_blocks() -> None:
    NAME = "90. difficulty_missing_estimate_blocks"
    try:
        service = DifficultyCalibrationService()
        report = service.calibrate(target_theta=0.5, difficulty_estimate=None)
        assert report.aligned is False, "missing estimate should fail-closed"
        assert report.predicted_difficulty is None
        assert report.delta is None
        assert report.predicted_band is None
        assert any("missing" in i.lower() for i in report.issues)
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 91. Difficulty calibrator invoked in LangGraph path
# ---------------------------------------------------------------------------

def test_difficulty_calibrator_invoked_in_langgraph() -> None:
    NAME = "91. difficulty_calibrator_invoked_in_langgraph"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        _, _mock_ret, _mock_planner, _mock_reranker, mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, mock_difficulty, mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            return_mocks=True,
        )

        mock_difficulty.calibrate.assert_called_once()
        call_kwargs = mock_difficulty.calibrate.call_args
        assert call_kwargs.kwargs["target_theta"] == 0.5
        assert call_kwargs.kwargs["difficulty_estimate"] == 0.4

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 92. Difficulty misalignment blocks persistence
# ---------------------------------------------------------------------------

def test_difficulty_misalignment_blocks_persistence() -> None:
    NAME = "92. difficulty_misalignment_blocks_persistence"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        bad_difficulty_report = DifficultyCalibrationReport(
            aligned=False, target_theta=1.0, predicted_difficulty=-1.5,
            delta=2.5, target_band="hard", predicted_band="easy",
            issues=["Delta 2.500 exceeds maximum 1.5"],
        )

        result, _mock_ret, _mock_planner, _mock_reranker, mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, mock_difficulty, mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            difficulty_return=bad_difficulty_report,
            return_mocks=True,
        )

        assert result is None, "difficulty misalignment should block persistence"
        mock_gen_svc.generate.assert_called()

        count = db.query(GeneratedQuestion).filter(
            GeneratedQuestion.source_exam_id == ids["exam_id"],
        ).count()
        assert count == 0, f"expected 0 persisted questions, got {count}"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 93. Calibration failure blocks persist
# ---------------------------------------------------------------------------

def test_difficulty_exception_blocks_persist() -> None:
    NAME = "93. difficulty_exception_blocks_persist"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result, _mock_ret, _mock_planner, _mock_reranker, mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, mock_difficulty, mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            difficulty_side_effect=RuntimeError("calibrator exploded"),
            return_mocks=True,
        )

        assert result is None, "difficulty exception should block persistence"

        count = db.query(GeneratedQuestion).filter(
            GeneratedQuestion.source_exam_id == ids["exam_id"],
        ).count()
        assert count == 0, f"expected 0 persisted questions, got {count}"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 94. Calibration report persisted in validation_report
# ---------------------------------------------------------------------------

def test_difficulty_report_in_validation_report() -> None:
    NAME = "94. difficulty_report_in_validation_report"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5, generated=0)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True, RAG_REVIEW_REQUIRED=False),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
        )

        assert result is not None, "graph should persist"
        vr = result.validation_report
        assert vr is not None, "validation_report must be set"
        assert "difficulty" in vr, "validation_report must contain difficulty"

        diff = vr["difficulty"]
        assert diff["aligned"] is True
        assert diff["target_theta"] == 0.5
        assert diff["predicted_difficulty"] == 0.4
        assert diff["delta"] is not None
        assert diff["target_band"] == "medium"
        assert diff["predicted_band"] == "medium"
        assert isinstance(diff["issues"], list)

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 95. Legacy path unchanged by difficulty calibrator
# ---------------------------------------------------------------------------

def test_legacy_path_unchanged_by_difficulty() -> None:
    NAME = "95. legacy_path_unchanged_by_difficulty"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        mock_run_rag_graph = MagicMock(return_value=None)
        mock_diff_cls = MagicMock()

        with patch(
            "app.services.langgraph_rag_workflow.run_rag_graph",
            mock_run_rag_graph,
        ), patch(
            "app.services.langgraph_rag_workflow.DifficultyCalibrationService",
            mock_diff_cls,
        ):
            result, mock_ret, _mock_planner, _mock_reranker, _mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_question_repair, _mock_artifact = _invoke(
                db, ids,
                _Settings(RAG_LANGGRAPH_ENABLED=False),
                chunks=chunks,
                gen=_fake_gen(),
                report=_valid_report(),
                langgraph=False,
                return_mocks=True,
            )

        assert result is not None, "legacy path returned None"

        # LangGraph entrypoint was never called
        mock_run_rag_graph.assert_not_called()

        # DifficultyCalibrationService was never instantiated
        mock_diff_cls.assert_not_called()

        # Legacy still uses retrieve directly
        mock_ret.retrieve.assert_called_once()

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 96. Missing difficulty_estimate blocks at node level (direct invocation)
# ---------------------------------------------------------------------------

def test_difficulty_missing_estimate_blocks_node() -> None:
    NAME = "96. difficulty_missing_estimate_blocks_node"
    try:
        from app.services.langgraph_rag_workflow import difficulty_calibrator

        gen_no_diff = GenerationOutput(
            question_text="What is the primary purpose of encryption?",
            options=[
                {"text": "Protect confidentiality", "is_correct": True},
                {"text": "Increase speed", "is_correct": False},
                {"text": "Compress files", "is_correct": False},
                {"text": "Generate randomness", "is_correct": False},
            ],
            explanation="Encryption converts plaintext to ciphertext.",
            difficulty_estimate=None,
        )

        state = {"gen_output": gen_no_diff, "theta": 0.5}
        result = difficulty_calibrator(state)

        assert "failure_reason" in result, "node should set failure_reason"
        assert "difficulty_report" in result, "node should return difficulty_report"
        report = result["difficulty_report"]
        assert report.aligned is False, "missing estimate should be blocked"
        assert report.predicted_difficulty is None
        assert report.delta is None
        assert report.target_band == "medium"
        assert report.predicted_band is None
        assert any("missing" in i.lower() for i in report.issues)
        assert "exception" not in result["failure_reason"].lower(), (
            "failure_reason should reflect calibration, not generic exception"
        )

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 97. Repair decision classifies distractor failure as repairable
# ---------------------------------------------------------------------------

def test_repair_decision_classifies_distractor() -> None:
    NAME = "97. repair_decision_classifies_distractor"
    try:
        from app.services.question_repair_service import QuestionRepairService
        svc = QuestionRepairService()
        report = svc.decide(
            failure_reason="Distractor validation failed: issues=['too similar']",
            repair_attempt_count=0,
        )
        assert report.repairable is True
        assert report.failure_type == "distractor"
        assert report.attempts_remaining == 0  # MAX=1, consumed 0, 1-0-1=0
        assert report.hint.get("target") == "distractors"
        assert "REPAIR INSTRUCTION" in report.hint.get("context_addendum", "")
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 98. Repair decision classifies difficulty failure as repairable
# ---------------------------------------------------------------------------

def test_repair_decision_classifies_difficulty() -> None:
    NAME = "98. repair_decision_classifies_difficulty"
    try:
        from app.services.question_repair_service import QuestionRepairService
        svc = QuestionRepairService()
        report = svc.decide(
            failure_reason="Difficulty misaligned: delta=2.5, issues=['too hard']",
            repair_attempt_count=0,
            target_theta=0.5,
            difficulty_signed_delta=2.0,  # predicted (2.5) - target (0.5) = +2.0 → too hard
        )
        assert report.repairable is True
        assert report.failure_type == "difficulty"
        assert report.hint.get("target") == "difficulty"
        # Positive signed delta → question too hard → lower theta (easier)
        assert report.hint.get("adjusted_theta") is not None
        assert float(report.hint["adjusted_theta"]) == 0.2, (
            f"expected 0.2 (0.5 - 0.3), got {report.hint['adjusted_theta']}"
        )
        assert "easier" in report.hint.get("context_addendum", "").lower()
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 99. Repair decision blocks evidence failure (terminal)
# ---------------------------------------------------------------------------

def test_repair_decision_blocks_evidence() -> None:
    NAME = "99. repair_decision_blocks_evidence"
    try:
        from app.services.question_repair_service import QuestionRepairService
        svc = QuestionRepairService()
        report = svc.decide(
            failure_reason="Evidence insufficient: low similarity",
            repair_attempt_count=0,
        )
        assert report.repairable is False
        assert report.failure_type == "evidence"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 100. Repair decision blocks grounding failure (terminal)
# ---------------------------------------------------------------------------

def test_repair_decision_blocks_grounding() -> None:
    NAME = "100. repair_decision_blocks_grounding"
    try:
        from app.services.question_repair_service import QuestionRepairService
        svc = QuestionRepairService()
        report = svc.decide(
            failure_reason="Grounding failed: score=0.05, issues=['unsupported']",
            repair_attempt_count=0,
        )
        assert report.repairable is False
        assert report.failure_type == "grounding"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 101. Repair decision blocks duplicate failure (terminal)
# ---------------------------------------------------------------------------

def test_repair_decision_blocks_duplicate() -> None:
    NAME = "101. repair_decision_blocks_duplicate"
    try:
        from app.services.question_repair_service import QuestionRepairService
        svc = QuestionRepairService()
        report = svc.decide(
            failure_reason="Duplicate detected: similarity=0.92, source=generated",
            repair_attempt_count=0,
        )
        assert report.repairable is False
        assert report.failure_type == "duplicate"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 102. Repair budget enforced — second attempt blocked
# ---------------------------------------------------------------------------

def test_repair_budget_enforced() -> None:
    NAME = "102. repair_budget_enforced"
    try:
        from app.services.question_repair_service import QuestionRepairService
        svc = QuestionRepairService()
        # First attempt — allowed
        r1 = svc.decide(
            failure_reason="Distractor validation failed",
            repair_attempt_count=0,
        )
        assert r1.repairable is True

        # Second attempt — budget exhausted
        r2 = svc.decide(
            failure_reason="Distractor validation failed",
            repair_attempt_count=1,
        )
        assert r2.repairable is False
        assert r2.attempts_remaining == 0
        assert any("exhausted" in i.lower() for i in r2.issues)
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 103. Repair hint adjusts theta for difficulty (easier direction)
# ---------------------------------------------------------------------------

def test_repair_hint_adjusts_theta_too_hard() -> None:
    NAME = "103. repair_hint_adjusts_theta_too_hard"
    try:
        from app.services.question_repair_service import QuestionRepairService
        svc = QuestionRepairService()
        # signed_delta > 0 → question too hard → lower theta (easier)
        report = svc.decide(
            failure_reason="Difficulty misaligned: delta=2.0",
            repair_attempt_count=0,
            target_theta=0.5,
            difficulty_signed_delta=2.0,  # predicted=2.5, target=0.5
        )
        assert report.repairable is True
        adjusted = float(report.hint["adjusted_theta"])
        assert adjusted == 0.2, (
            f"expected exact 0.2 (target 0.5 - 0.3 bounded nudge), got {adjusted}"
        )
        assert "easier" in report.hint["context_addendum"].lower()
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 103b. Repair hint adjusts theta for difficulty (too-easy direction)
# ---------------------------------------------------------------------------

def test_repair_hint_adjusts_theta_too_easy() -> None:
    NAME = "103b. repair_hint_adjusts_theta_too_easy"
    try:
        from app.services.question_repair_service import QuestionRepairService
        svc = QuestionRepairService()
        # signed_delta < 0 → question too easy → raise theta (harder)
        report = svc.decide(
            failure_reason="Difficulty misaligned: delta=2.0",
            repair_attempt_count=0,
            target_theta=0.5,
            difficulty_signed_delta=-2.0,  # predicted=-1.5, target=0.5
        )
        assert report.repairable is True
        adjusted = float(report.hint["adjusted_theta"])
        assert adjusted == 0.8, (
            f"expected exact 0.8 (target 0.5 + 0.3 bounded nudge), got {adjusted}"
        )
        assert "harder" in report.hint["context_addendum"].lower()
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# 104. Repair triggered on distractor failure — re-enters generate
# ---------------------------------------------------------------------------

def test_repair_triggered_on_distractor_failure() -> None:
    NAME = "104. repair_triggered_on_distractor_failure"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        bad_distractor_report = DistractorReport(
            valid=False, distinct_distractors=False,
            separated_from_correct=True, meaningful_distractors=True,
            issues=["Distractors 1 and 2 are near-duplicates (Jaccard=0.90)"],
        )
        good_distractor_report = DistractorReport(
            valid=True, distinct_distractors=True,
            separated_from_correct=True, meaningful_distractors=True,
            issues=[],
        )
        repair_report = RepairReport(
            repairable=True, attempts_remaining=0,
            failure_type="distractor",
            hint={"target": "distractors", "context_addendum": "Fix distractors", "adjusted_theta": None},
        )

        result, _mock_ret, _mock_planner, _mock_reranker, mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, mock_distractor, _mock_diff, mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            distractor_side_effect=[bad_distractor_report, good_distractor_report],
            question_repair_return=repair_report,
            return_mocks=True,
        )

        assert result is not None, "should persist after repair"
        # Distractor validator should have been called twice (first fail, then pass after repair)
        assert mock_distractor.validate.call_count == 2, (
            f"expected 2 distractor calls, got {mock_distractor.validate.call_count}"
        )
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 105. Second failure after repair aborts safely
# ---------------------------------------------------------------------------

def test_second_failure_after_repair_aborts() -> None:
    NAME = "105. second_failure_after_repair_aborts"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        bad_distractor_report = DistractorReport(
            valid=False, distinct_distractors=False,
            separated_from_correct=True, meaningful_distractors=True,
            issues=["near-duplicates"],
        )
        # First call: repairable, second call: budget exhausted
        repair_allowed = RepairReport(
            repairable=True, attempts_remaining=0,
            failure_type="distractor",
            hint={"target": "distractors", "context_addendum": "Fix", "adjusted_theta": None},
        )
        repair_blocked = RepairReport(
            repairable=False, attempts_remaining=0,
            failure_type="distractor",
            issues=["Repair budget exhausted: 1/1"],
        )

        # Both distractor calls return bad; repair allows first, blocks second
        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            distractor_side_effect=[bad_distractor_report, bad_distractor_report],
            question_repair_side_effect=[repair_allowed, repair_blocked],
        )

        assert result is None, "should abort after second failure"
        count = db.query(GeneratedQuestion).filter(
            GeneratedQuestion.source_exam_id == ids["exam_id"],
        ).count()
        assert count == 0, f"expected 0 persisted, got {count}"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 106. Non-repairable failure does not enter repair loop
# ---------------------------------------------------------------------------

def test_non_repairable_failure_skips_repair() -> None:
    NAME = "106. non_repairable_failure_skips_repair"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        grounding_fail = GroundingReport(
            grounded=False, question_supported=False,
            answer_supported=False, explanation_supported=False,
            support_score=0.02, issues=["no lexical support"],
        )

        result, _mock_ret, _mock_planner, _mock_reranker, _mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, mock_question_repair, _mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            grounding_return=grounding_fail,
            return_mocks=True,
        )

        assert result is None, "grounding failure should abort"
        # repair_decision should NOT have been called — grounding is terminal
        mock_question_repair.decide.assert_not_called()
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 107. Max repair count enforced — second repair attempt blocked
# ---------------------------------------------------------------------------

def test_max_repair_count_enforced() -> None:
    NAME = "107. max_repair_count_enforced"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        bad_difficulty = DifficultyCalibrationReport(
            aligned=False, target_theta=1.0, predicted_difficulty=-1.5,
            delta=2.5, target_band="hard", predicted_band="easy",
            issues=["Delta 2.500 exceeds maximum 1.5"],
        )
        # First call: repairable, second call: budget exhausted
        repair_allowed = RepairReport(
            repairable=True, attempts_remaining=0,
            failure_type="difficulty",
            hint={"target": "difficulty", "context_addendum": "Fix", "adjusted_theta": 0.8},
        )
        repair_blocked = RepairReport(
            repairable=False, attempts_remaining=0,
            failure_type="difficulty",
            issues=["Repair budget exhausted: 1/1"],
        )

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            difficulty_return=bad_difficulty,
            question_repair_side_effect=[repair_allowed, repair_blocked],
        )

        assert result is None, "should abort after budget exhausted"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 108. Legacy path unchanged by repair decision
# ---------------------------------------------------------------------------

def test_legacy_path_unchanged_by_repair() -> None:
    NAME = "108. legacy_path_unchanged_by_repair"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        mock_run_rag_graph = MagicMock(return_value=None)
        mock_repair_cls = MagicMock()

        with patch(
            "app.services.langgraph_rag_workflow.run_rag_graph",
            mock_run_rag_graph,
        ), patch(
            "app.services.langgraph_rag_workflow.QuestionRepairService",
            mock_repair_cls,
        ):
            result, mock_ret, _mock_planner, _mock_reranker, _mock_gen_svc, _mock_comp, _mock_repair, _mock_dedup, _mock_grounding, _mock_distractor, _mock_difficulty, _mock_qr, _mock_artifact = _invoke(
                db, ids,
                _Settings(RAG_LANGGRAPH_ENABLED=False),
                chunks=chunks,
                gen=_fake_gen(),
                report=_valid_report(),
                langgraph=False,
                return_mocks=True,
            )

        assert result is not None, "legacy path returned None"
        mock_run_rag_graph.assert_not_called()
        mock_repair_cls.assert_not_called()
        mock_ret.retrieve.assert_called_once()
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 109. Repair metadata persisted in validation_report
# ---------------------------------------------------------------------------

def test_repair_metadata_persisted() -> None:
    NAME = "109. repair_metadata_persisted"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5, generated=0)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        bad_distractor = DistractorReport(
            valid=False, distinct_distractors=False,
            separated_from_correct=True, meaningful_distractors=True,
            issues=["near-duplicates"],
        )
        good_distractor = DistractorReport(
            valid=True, distinct_distractors=True,
            separated_from_correct=True, meaningful_distractors=True,
            issues=[],
        )
        repair_report = RepairReport(
            repairable=True, attempts_remaining=0,
            failure_type="distractor",
            hint={"target": "distractors", "context_addendum": "Fix", "adjusted_theta": None},
        )

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True, RAG_REVIEW_REQUIRED=False),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
            langgraph=True,
            distractor_side_effect=[bad_distractor, good_distractor],
            question_repair_return=repair_report,
        )

        assert result is not None, "should persist after repair"
        vr = result.validation_report
        assert vr is not None
        assert "repair" in vr, "validation_report must contain repair metadata"
        assert vr["repair"]["attempt_count"] == 1
        assert vr["repair"]["final_hint_target"] == "distractors"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 110. Missing difficulty_estimate blocks in real LangGraph path
# ---------------------------------------------------------------------------

def test_missing_difficulty_blocks_langgraph_path() -> None:
    NAME = "110. missing_difficulty_blocks_langgraph_path"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        # Simulate LLM returning JSON without difficulty_estimate
        gen_no_diff = GenerationOutput(
            question_text="What is the primary purpose of encryption?",
            options=[
                {"text": "Protect confidentiality", "is_correct": True},
                {"text": "Increase speed", "is_correct": False},
                {"text": "Compress files", "is_correct": False},
                {"text": "Generate randomness", "is_correct": False},
            ],
            explanation="Encryption converts plaintext to ciphertext.",
            difficulty_estimate=None,
        )

        # Make the mock enforce real fail-closed policy for None
        def _calibrate_enforce(**kwargs):
            if kwargs.get("difficulty_estimate") is None:
                raise ValueError("difficulty_estimate must be provided")
            from app.services.difficulty_calibration_service import (
                DifficultyCalibrationService as _Real,
            )
            return _Real().calibrate(**kwargs)

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=gen_no_diff,
            report=_valid_report(),
            langgraph=True,
            difficulty_side_effect=_calibrate_enforce,
        )

        assert result is None, "missing difficulty_estimate should block persistence"
        count = db.query(GeneratedQuestion).filter(
            GeneratedQuestion.source_exam_id == ids["exam_id"],
        ).count()
        assert count == 0, f"expected 0 persisted, got {count}"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# 111. Real JSON parsing path — missing difficulty_estimate in LLM output
# ---------------------------------------------------------------------------

def test_real_json_parsing_missing_difficulty_blocks() -> None:
    """Exercise the actual GeneratedQuestionService.generate() parsing path.

    Mocks the LLM client to return raw JSON *without* difficulty_estimate,
    verifies that generate() returns GenerationOutput(difficulty_estimate=None),
    then feeds that through the LangGraph path with a real DifficultyCalibrationService
    to prove it blocks persistence (fail-closed).
    """
    NAME = "111. real_json_parsing_missing_difficulty_blocks"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        # LLM returns valid JSON but omits difficulty_estimate entirely
        raw_json = (
            '{"question_text": "What is the primary purpose of encryption?",'
            ' "options": ['
            '{"text": "Protect confidentiality", "is_correct": true},'
            '{"text": "Increase speed", "is_correct": false},'
            '{"text": "Compress files", "is_correct": false},'
            '{"text": "Generate randomness", "is_correct": false}'
            '],'
            ' "explanation": "Encryption converts plaintext to ciphertext."}'
        )

        # Step 1: Prove the real generate() parsing path produces None
        from app.services.generated_question_service import (
            GeneratedQuestionService,
            GenerationInput,
        )

        mock_llm = MagicMock()
        mock_llm.is_available = True
        mock_llm.generate_chat.return_value = raw_json

        with patch(
            "app.services.generated_question_service.LLMClient",
            return_value=mock_llm,
        ):
            gen_svc = GeneratedQuestionService(db)
            gen_output = gen_svc.generate(
                GenerationInput(
                    topic_id=ids["topic_id"],
                    topic_name="Cryptography",
                    theta=0.5,
                    recent_streak=5,
                    avg_theta=0.5,
                    retrieved_chunks=chunks,
                )
            )

        assert gen_output is not None, "generate() should return an output"
        assert gen_output.difficulty_estimate is None, (
            f"expected None from real parsing, got {gen_output.difficulty_estimate}"
        )

        # Step 2: Feed through LangGraph path — enforce real calibrator policy
        def _calibrate_enforce(**kwargs):
            if kwargs.get("difficulty_estimate") is None:
                raise ValueError("difficulty_estimate must be provided")
            from app.services.difficulty_calibration_service import (
                DifficultyCalibrationService as _Real,
            )
            return _Real().calibrate(**kwargs)

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=gen_output,
            report=_valid_report(),
            langgraph=True,
            difficulty_side_effect=_calibrate_enforce,
        )

        assert result is None, (
            "LangGraph path should block persistence when difficulty_estimate is None"
        )
        count = db.query(GeneratedQuestion).filter(
            GeneratedQuestion.source_exam_id == ids["exam_id"],
        ).count()
        assert count == 0, f"expected 0 persisted, got {count}"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# Phase 2.12 — Artifact validation (citations + distractor rationale)
# ---------------------------------------------------------------------------

def test_artifact_valid_citations_and_rationale_persists() -> None:
    """Valid citations and rationale should persist the question."""
    NAME = "112. artifact_valid_citations_and_rationale_persists"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])
        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen_with_artifacts(),
            report=_valid_report(),
            langgraph=True,
        )
        assert result is not None, "should persist with valid artifacts"
        assert isinstance(result, GeneratedQuestion)
        vr = result.validation_report
        assert vr is not None
        assert vr.get("artifact", {}).get("citations_valid") is True
        assert vr.get("artifact", {}).get("rationale_valid") is True
        assert vr.get("evidence_citations") is not None
        assert vr.get("distractor_rationale") is not None
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()


def test_artifact_missing_citations_blocks() -> None:
    """Missing citations when required should block persistence."""
    NAME = "113. artifact_missing_citations_blocks"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])
        gen_no_citations = GenerationOutput(
            question_text="What is the primary purpose of encryption?",
            options=[
                {"text": "Protect confidentiality", "is_correct": True},
                {"text": "Increase speed", "is_correct": False},
                {"text": "Compress files", "is_correct": False},
                {"text": "Generate randomness", "is_correct": False},
            ],
            explanation="Encryption converts plaintext to ciphertext.",
            difficulty_estimate=0.4,
            evidence_citations=None,
            distractor_rationale={"1": "Speed is unrelated.", "2": "Compression is separate.", "3": "Randomness is for keys."},
        )
        # Enforce real artifact validator
        def _artifact_enforce(**kwargs):
            from app.services.generated_artifact_validation_service import (
                GeneratedArtifactValidationService as _Real,
            )
            settings = _Settings(RAG_LANGGRAPH_ENABLED=True)
            return _Real(
                require_citations=settings.GENERATED_MCQ_REQUIRE_CITATIONS,
                require_rationale=settings.GENERATED_MCQ_REQUIRE_DISTRACTOR_RATIONALE,
            ).validate(**kwargs)

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=gen_no_citations,
            report=_valid_report(),
            langgraph=True,
            artifact_side_effect=_artifact_enforce,
        )
        assert result is None, "missing citations should block persistence"
        count = db.query(GeneratedQuestion).filter(
            GeneratedQuestion.source_exam_id == ids["exam_id"],
        ).count()
        assert count == 0, f"expected 0 persisted, got {count}"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()


def test_artifact_empty_citations_blocks() -> None:
    """Empty citations list should block persistence."""
    NAME = "114. artifact_empty_citations_blocks"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])
        gen_empty_citations = GenerationOutput(
            question_text="What is the primary purpose of encryption?",
            options=[
                {"text": "Protect confidentiality", "is_correct": True},
                {"text": "Increase speed", "is_correct": False},
                {"text": "Compress files", "is_correct": False},
                {"text": "Generate randomness", "is_correct": False},
            ],
            explanation="Encryption converts plaintext to ciphertext.",
            difficulty_estimate=0.4,
            evidence_citations=[],
            distractor_rationale={"1": "Speed is unrelated.", "2": "Compression is separate.", "3": "Randomness is for keys."},
        )
        def _artifact_enforce(**kwargs):
            from app.services.generated_artifact_validation_service import (
                GeneratedArtifactValidationService as _Real,
            )
            return _Real(require_citations=True, require_rationale=True).validate(**kwargs)

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=gen_empty_citations,
            report=_valid_report(),
            langgraph=True,
            artifact_side_effect=_artifact_enforce,
        )
        assert result is None, "empty citations should block persistence"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()


def test_artifact_short_citations_blocks() -> None:
    """Citations shorter than 5 chars should block persistence."""
    NAME = "115. artifact_short_citations_blocks"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])
        gen_short = GenerationOutput(
            question_text="What is the primary purpose of encryption?",
            options=[
                {"text": "Protect confidentiality", "is_correct": True},
                {"text": "Increase speed", "is_correct": False},
                {"text": "Compress files", "is_correct": False},
                {"text": "Generate randomness", "is_correct": False},
            ],
            explanation="Encryption converts plaintext to ciphertext.",
            difficulty_estimate=0.4,
            evidence_citations=["abc"],
            distractor_rationale={"1": "Speed is unrelated.", "2": "Compression is separate.", "3": "Randomness is for keys."},
        )
        def _artifact_enforce(**kwargs):
            from app.services.generated_artifact_validation_service import (
                GeneratedArtifactValidationService as _Real,
            )
            return _Real(require_citations=True, require_rationale=True).validate(**kwargs)

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=gen_short,
            report=_valid_report(),
            langgraph=True,
            artifact_side_effect=_artifact_enforce,
        )
        assert result is None, "short citations should block persistence"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()


def test_artifact_fabricated_citations_blocks() -> None:
    """Citations with no word overlap with evidence should block persistence."""
    NAME = "116. artifact_fabricated_citations_blocks"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])
        gen_fabricated = GenerationOutput(
            question_text="What is the primary purpose of encryption?",
            options=[
                {"text": "Protect confidentiality", "is_correct": True},
                {"text": "Increase speed", "is_correct": False},
                {"text": "Compress files", "is_correct": False},
                {"text": "Generate randomness", "is_correct": False},
            ],
            explanation="Encryption converts plaintext to ciphertext.",
            difficulty_estimate=0.4,
            evidence_citations=["Quantum entanglement enables faster-than-light communication"],
            distractor_rationale={"1": "Speed is unrelated.", "2": "Compression is separate.", "3": "Randomness is for keys."},
        )
        def _artifact_enforce(**kwargs):
            from app.services.generated_artifact_validation_service import (
                GeneratedArtifactValidationService as _Real,
            )
            return _Real(require_citations=True, require_rationale=True).validate(**kwargs)

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=gen_fabricated,
            report=_valid_report(),
            langgraph=True,
            artifact_side_effect=_artifact_enforce,
        )
        assert result is None, "fabricated citations should block persistence"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()


def test_artifact_missing_rationale_blocks() -> None:
    """Missing distractor_rationale when required should block persistence."""
    NAME = "117. artifact_missing_rationale_blocks"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])
        gen_no_rationale = GenerationOutput(
            question_text="What is the primary purpose of encryption?",
            options=[
                {"text": "Protect confidentiality", "is_correct": True},
                {"text": "Increase speed", "is_correct": False},
                {"text": "Compress files", "is_correct": False},
                {"text": "Generate randomness", "is_correct": False},
            ],
            explanation="Encryption converts plaintext to ciphertext.",
            difficulty_estimate=0.4,
            evidence_citations=["Encryption converts plaintext to ciphertext for protection"],
            distractor_rationale=None,
        )
        def _artifact_enforce(**kwargs):
            from app.services.generated_artifact_validation_service import (
                GeneratedArtifactValidationService as _Real,
            )
            return _Real(require_citations=True, require_rationale=True).validate(**kwargs)

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=gen_no_rationale,
            report=_valid_report(),
            langgraph=True,
            artifact_side_effect=_artifact_enforce,
        )
        assert result is None, "missing rationale should block persistence"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()


def test_artifact_incomplete_rationale_blocks() -> None:
    """Rationale missing a wrong option should block persistence."""
    NAME = "118. artifact_incomplete_rationale_blocks"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])
        gen_incomplete = GenerationOutput(
            question_text="What is the primary purpose of encryption?",
            options=[
                {"text": "Protect confidentiality", "is_correct": True},
                {"text": "Increase speed", "is_correct": False},
                {"text": "Compress files", "is_correct": False},
                {"text": "Generate randomness", "is_correct": False},
            ],
            explanation="Encryption converts plaintext to ciphertext.",
            difficulty_estimate=0.4,
            evidence_citations=["Encryption converts plaintext to ciphertext for protection"],
            distractor_rationale={"1": "Speed is unrelated."},  # missing indices 2 and 3
        )
        def _artifact_enforce(**kwargs):
            from app.services.generated_artifact_validation_service import (
                GeneratedArtifactValidationService as _Real,
            )
            return _Real(require_citations=True, require_rationale=True).validate(**kwargs)

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=gen_incomplete,
            report=_valid_report(),
            langgraph=True,
            artifact_side_effect=_artifact_enforce,
        )
        assert result is None, "incomplete rationale should block persistence"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()


def test_artifact_rationale_with_correct_entry_blocks() -> None:
    """Rationale that includes the correct option index should block."""
    NAME = "119. artifact_rationale_with_correct_entry_blocks"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])
        gen_bad = GenerationOutput(
            question_text="What is the primary purpose of encryption?",
            options=[
                {"text": "Protect confidentiality", "is_correct": True},
                {"text": "Increase speed", "is_correct": False},
                {"text": "Compress files", "is_correct": False},
                {"text": "Generate randomness", "is_correct": False},
            ],
            explanation="Encryption converts plaintext to ciphertext.",
            difficulty_estimate=0.4,
            evidence_citations=["Encryption converts plaintext to ciphertext for protection"],
            distractor_rationale={
                "0": "This is actually correct.",  # wrong: correct answer has rationale
                "1": "Speed is unrelated.",
                "2": "Compression is separate.",
                "3": "Randomness is for keys.",
            },
        )
        def _artifact_enforce(**kwargs):
            from app.services.generated_artifact_validation_service import (
                GeneratedArtifactValidationService as _Real,
            )
            return _Real(require_citations=True, require_rationale=True).validate(**kwargs)

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=gen_bad,
            report=_valid_report(),
            langgraph=True,
            artifact_side_effect=_artifact_enforce,
        )
        assert result is None, "rationale with correct entry should block"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()


def test_artifact_empty_rationale_text_blocks() -> None:
    """Empty rationale text values should block persistence."""
    NAME = "120. artifact_empty_rationale_text_blocks"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])
        gen_empty_text = GenerationOutput(
            question_text="What is the primary purpose of encryption?",
            options=[
                {"text": "Protect confidentiality", "is_correct": True},
                {"text": "Increase speed", "is_correct": False},
                {"text": "Compress files", "is_correct": False},
                {"text": "Generate randomness", "is_correct": False},
            ],
            explanation="Encryption converts plaintext to ciphertext.",
            difficulty_estimate=0.4,
            evidence_citations=["Encryption converts plaintext to ciphertext for protection"],
            distractor_rationale={"1": "", "2": "  ", "3": ""},
        )
        def _artifact_enforce(**kwargs):
            from app.services.generated_artifact_validation_service import (
                GeneratedArtifactValidationService as _Real,
            )
            return _Real(require_citations=True, require_rationale=True).validate(**kwargs)

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=gen_empty_text,
            report=_valid_report(),
            langgraph=True,
            artifact_side_effect=_artifact_enforce,
        )
        assert result is None, "empty rationale text should block"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()


def test_artifact_config_off_citations_skipped() -> None:
    """When GENERATED_MCQ_REQUIRE_CITATIONS=False, missing citations pass."""
    NAME = "121. artifact_config_off_citations_skipped"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])
        gen_no_citations = GenerationOutput(
            question_text="What is the primary purpose of encryption?",
            options=[
                {"text": "Protect confidentiality", "is_correct": True},
                {"text": "Increase speed", "is_correct": False},
                {"text": "Compress files", "is_correct": False},
                {"text": "Generate randomness", "is_correct": False},
            ],
            explanation="Encryption converts plaintext to ciphertext.",
            difficulty_estimate=0.4,
            evidence_citations=None,
            distractor_rationale={"1": "Speed is unrelated.", "2": "Compression is separate.", "3": "Randomness is for keys."},
        )
        result = _invoke(
            db, ids,
            _Settings(
                RAG_LANGGRAPH_ENABLED=True,
                GENERATED_MCQ_REQUIRE_CITATIONS=False,
            ),
            chunks=chunks,
            gen=gen_no_citations,
            report=_valid_report(),
            langgraph=True,
        )
        assert result is not None, "config-off citations should not block"
        assert isinstance(result, GeneratedQuestion)
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()


def test_artifact_config_off_rationale_skipped() -> None:
    """When GENERATED_MCQ_REQUIRE_DISTRACTOR_RATIONALE=False, missing rationale passes."""
    NAME = "122. artifact_config_off_rationale_skipped"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])
        gen_no_rationale = GenerationOutput(
            question_text="What is the primary purpose of encryption?",
            options=[
                {"text": "Protect confidentiality", "is_correct": True},
                {"text": "Increase speed", "is_correct": False},
                {"text": "Compress files", "is_correct": False},
                {"text": "Generate randomness", "is_correct": False},
            ],
            explanation="Encryption converts plaintext to ciphertext.",
            difficulty_estimate=0.4,
            evidence_citations=["Encryption converts plaintext to ciphertext for protection"],
            distractor_rationale=None,
        )
        result = _invoke(
            db, ids,
            _Settings(
                RAG_LANGGRAPH_ENABLED=True,
                GENERATED_MCQ_REQUIRE_DISTRACTOR_RATIONALE=False,
            ),
            chunks=chunks,
            gen=gen_no_rationale,
            report=_valid_report(),
            langgraph=True,
        )
        assert result is not None, "config-off rationale should not block"
        assert isinstance(result, GeneratedQuestion)
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()


def test_artifact_both_config_off() -> None:
    """When both config flags are False, missing citations AND rationale pass."""
    NAME = "123. artifact_both_config_off"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])
        gen_bare = GenerationOutput(
            question_text="What is the primary purpose of encryption?",
            options=[
                {"text": "Protect confidentiality", "is_correct": True},
                {"text": "Increase speed", "is_correct": False},
                {"text": "Compress files", "is_correct": False},
                {"text": "Generate randomness", "is_correct": False},
            ],
            explanation="Encryption converts plaintext to ciphertext.",
            difficulty_estimate=0.4,
            evidence_citations=None,
            distractor_rationale=None,
        )
        result = _invoke(
            db, ids,
            _Settings(
                RAG_LANGGRAPH_ENABLED=True,
                GENERATED_MCQ_REQUIRE_CITATIONS=False,
                GENERATED_MCQ_REQUIRE_DISTRACTOR_RATIONALE=False,
            ),
            chunks=chunks,
            gen=gen_bare,
            report=_valid_report(),
            langgraph=True,
        )
        assert result is not None, "both config-off should not block"
        assert isinstance(result, GeneratedQuestion)
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()


def test_legacy_path_unchanged_by_artifact() -> None:
    """Legacy path should not be affected by artifact validation."""
    NAME = "124. legacy_path_unchanged_by_artifact"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])
        # _fake_gen has no citations/rationale — legacy path should still work
        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=False),
            chunks=chunks,
            gen=_fake_gen(),
            report=_valid_report(),
        )
        assert result is not None, "legacy path should persist"
        assert isinstance(result, GeneratedQuestion)
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()


def test_artifact_report_in_validation_report() -> None:
    """Artifact metadata should be persisted in validation_report."""
    NAME = "125. artifact_report_in_validation_report"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])
        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen_with_artifacts(),
            report=_valid_report(),
            langgraph=True,
        )
        assert result is not None, "should persist"
        vr = result.validation_report
        assert vr is not None
        # artifact sub-dict present
        art = vr.get("artifact")
        assert art is not None, "artifact sub-dict missing from validation_report"
        assert art["citations_valid"] is True
        assert art["rationale_valid"] is True
        assert art["issues"] == []
        # raw artifacts stored
        assert vr.get("evidence_citations") is not None
        assert len(vr["evidence_citations"]) == 2
        assert vr.get("distractor_rationale") is not None
        assert len(vr["distractor_rationale"]) == 3
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()


def test_artifact_failure_blocks_persistence() -> None:
    """Artifact validation failure in LangGraph should not persist a question."""
    NAME = "126. artifact_failure_blocks_persistence"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])

        def _artifact_block(**kwargs):
            return ArtifactValidationReport(
                citations_valid=False,
                rationale_valid=False,
                issues=["citations required but missing", "rationale required but missing"],
            )

        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen(),  # no citations/rationale
            report=_valid_report(),
            langgraph=True,
            artifact_side_effect=_artifact_block,
        )
        assert result is None, "artifact failure should block persistence"
        count = db.query(GeneratedQuestion).filter(
            GeneratedQuestion.source_exam_id == ids["exam_id"],
        ).count()
        assert count == 0, f"expected 0 persisted, got {count}"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()


def test_artifact_validator_invoked_in_langgraph() -> None:
    """artifact_validator should be called in the LangGraph path."""
    NAME = "127. artifact_validator_invoked_in_langgraph"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])
        result, _, _, _, _, _, _, _, _, _, _, _, mock_artifact = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen_with_artifacts(),
            report=_valid_report(),
            langgraph=True,
            return_mocks=True,
        )
        assert result is not None, "should persist"
        mock_artifact.validate.assert_called_once()
        call_kwargs = mock_artifact.validate.call_args[1]
        assert "evidence_citations" in call_kwargs
        assert "distractor_rationale" in call_kwargs
        assert "options" in call_kwargs
        assert "retrieved_chunks" in call_kwargs
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()


def test_artifact_exception_blocks_persist() -> None:
    """Artifact validator exception should block persistence (fail-closed)."""
    NAME = "128. artifact_exception_blocks_persist"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])
        result = _invoke(
            db, ids,
            _Settings(RAG_LANGGRAPH_ENABLED=True),
            chunks=chunks,
            gen=_fake_gen_with_artifacts(),
            report=_valid_report(),
            langgraph=True,
            artifact_side_effect=RuntimeError("artifact service exploded"),
        )
        assert result is None, "exception should block persistence"
        count = db.query(GeneratedQuestion).filter(
            GeneratedQuestion.source_exam_id == ids["exam_id"],
        ).count()
        assert count == 0, f"expected 0 persisted, got {count}"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()


def test_artifact_partial_invalid_citations_blocks() -> None:
    """Any malformed citation entry should block, even if another citation is valid."""
    NAME = "129. artifact_partial_invalid_citations_blocks"
    db = SessionLocal()
    try:
        ids = _setup(db)
        _set_streak(db, ids, streak=5)
        chunks = _fake_chunks(ids["topic_id"], chunk_ids=ids["chunk_ids"], document_id=ids["document_id"])
        gen_bad = GenerationOutput(
            question_text="What is the primary purpose of encryption?",
            options=[
                {"text": "Protect confidentiality", "is_correct": True},
                {"text": "Increase speed", "is_correct": False},
                {"text": "Compress files", "is_correct": False},
                {"text": "Generate randomness", "is_correct": False},
            ],
            explanation="Encryption converts plaintext to ciphertext.",
            difficulty_estimate=0.4,
            evidence_citations=["Encryption converts plaintext to ciphertext for protection", "bad"],
            distractor_rationale={"1": "Speed is unrelated.", "2": "Compression is separate.", "3": "Randomness is for keys."},
        )
        def _artifact_enforce(**kwargs):
            from app.services.generated_artifact_validation_service import GeneratedArtifactValidationService as _Real
            return _Real(require_citations=True, require_rationale=True).validate(**kwargs)
        result = _invoke(db, ids, _Settings(RAG_LANGGRAPH_ENABLED=True), chunks=chunks, gen=gen_bad, report=_valid_report(), langgraph=True, artifact_side_effect=_artifact_enforce)
        assert result is None, "partially invalid citations should block persistence"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()


def test_generation_preserves_non_string_rationale_for_validation() -> None:
    """Generation parsing should preserve malformed rationale values for downstream validation."""
    NAME = "130. generation_preserves_non_string_rationale_for_validation"
    try:
        from app.services.generated_question_service import GeneratedQuestionService, GenerationInput
        fake_llm = MagicMock()
        fake_llm.is_available = True
        fake_llm.generate_chat.return_value = '{"question_text":"What is the primary purpose of encryption?","options":[{"text":"Protect confidentiality","is_correct":true},{"text":"Increase speed","is_correct":false},{"text":"Compress files","is_correct":false},{"text":"Generate randomness","is_correct":false}],"explanation":"Encryption converts plaintext to ciphertext.","difficulty_estimate":0.4,"evidence_citations":["Encryption converts plaintext to ciphertext for protection"],"distractor_rationale":{"1":null,"2":"Compression is separate.","3":"Randomness is for keys."}}'
        with patch("app.services.generated_question_service.get_settings", return_value=_Settings()), patch("app.services.generated_question_service.LLMClient", return_value=fake_llm):
            svc = GeneratedQuestionService(MagicMock())
            out = svc.generate(GenerationInput(topic_id=1, topic_name="Cryptography", theta=0.0, recent_streak=5, avg_theta=0.0, retrieved_chunks=_fake_chunks(1)))
        assert out is not None, "generation should still return parsed output"
        assert out.distractor_rationale is not None
        assert out.distractor_rationale["1"] is None, "non-string rationale values should not be coerced to strings"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# =========================================================================
# Fix-plan verification tests
# =========================================================================

# ---------------------------------------------------------------------------
# P0.1 — Streak correctness: before_order_index excludes current answer
# ---------------------------------------------------------------------------

def test_streak_before_order_index() -> None:
    """record_answer(before_order_index=N) looks at responses with
    order_index < N, so the just-inserted answer is not treated as
    'previous topic'."""
    NAME = "F1. streak_before_order_index"
    db = SessionLocal()
    try:
        from app.models import AdaptiveExamResponse, Question
        from app.models.enums import DifficultyLevel, CognitiveLevel
        from app.services.topic_streak_service import TopicStreakService

        ids = _setup(db)

        # Create two topics — A and B
        topic_a = Topic(phase_id=ids["phase_id"], name=f"__TopicA_{uuid.uuid4().hex[:6]}__", description="tA")
        db.add(topic_a)
        db.flush()
        sub_a = SubTopic(topic_id=topic_a.id, name=f"__SubA__", description="sA")
        db.add(sub_a)
        db.flush()

        topic_b = Topic(phase_id=ids["phase_id"], name=f"__TopicB_{uuid.uuid4().hex[:6]}__", description="tB")
        db.add(topic_b)
        db.flush()
        sub_b = SubTopic(topic_id=topic_b.id, name=f"__SubB__", description="sB")
        db.add(sub_b)
        db.flush()

        # Create separate questions for each response slot (unique constraint requires distinct question_ids)
        questions = []
        for i in range(3):
            sub = sub_a if i < 2 else sub_b  # first two → topic A, third → topic B
            q = Question(
                text=f"P01_q_{i}_{uuid.uuid4().hex[:6]}", difficulty=DifficultyLevel.Medium,
                cognitive_level=CognitiveLevel.Application, question_type="SingleChoice",
                subtopic_id=sub.id, is_active=True,
            )
            db.add(q)
            db.flush()
            questions.append(q)

        q_a1, q_a2, q_b = questions

        # Insert responses: A (order=0), A (order=1), B (order=2)
        for oi, q in [(0, q_a1), (1, q_a2), (2, q_b)]:
            resp = AdaptiveExamResponse(
                adaptive_exam_id=ids["exam_id"], question_id=q.id,
                order_index=oi, is_correct=True, theta_before=0.0, theta_after=0.0,
            )
            db.add(resp)
        db.flush()

        ts = TopicStreakService(db, ids["student_id"])

        # Record answer for topic A with before_order_index=3 (next slot).
        # The most recent response with order_index < 3 is B (order=2),
        # so the previous topic is B — NOT A.
        info = ts.record_answer(
            ids["exam_id"], topic_a.id, 0.0, before_order_index=3,
        )
        # streak should be 1 (B broke the streak, so A starts fresh)
        assert info.current_streak == 1, (
            f"expected streak=1 after A→A→B→A, got {info.current_streak}"
        )

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# P0.2 — Difficulty map uses IRT logit scale
# ---------------------------------------------------------------------------

def test_difficulty_map_logit_scale() -> None:
    """_map_difficulty_estimate uses IRT logit scale: -0.5=Easy, +0.5=Hard."""
    NAME = "F2. difficulty_map_logit_scale"
    try:
        from app.services.adaptive_exam_service import _map_difficulty_estimate
        from app.models.enums import DifficultyLevel

        assert _map_difficulty_estimate(None) == DifficultyLevel.Medium
        assert _map_difficulty_estimate(-1.0) == DifficultyLevel.Easy
        assert _map_difficulty_estimate(-0.6) == DifficultyLevel.Easy
        assert _map_difficulty_estimate(-0.5) == DifficultyLevel.Medium  # boundary is strict >
        assert _map_difficulty_estimate(0.0) == DifficultyLevel.Medium
        assert _map_difficulty_estimate(0.4) == DifficultyLevel.Medium
        assert _map_difficulty_estimate(0.5) == DifficultyLevel.Medium  # boundary is strict >
        assert _map_difficulty_estimate(0.6) == DifficultyLevel.Hard
        assert _map_difficulty_estimate(1.5) == DifficultyLevel.Hard

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# P1.4 — failure_code routes repair decision (stable code preferred)
# ---------------------------------------------------------------------------

def test_repair_decision_uses_failure_code() -> None:
    """decide() uses failure_code for classification when available,
    independent of failure_reason wording."""
    NAME = "F3. repair_decision_uses_failure_code"
    try:
        from app.services.question_repair_service import QuestionRepairService
        svc = QuestionRepairService()

        # failure_code='distractor_failed' should classify as 'distractor'
        # even if failure_reason says something else entirely
        r1 = svc.decide(
            failure_reason="something went wrong",
            repair_attempt_count=0,
            failure_code="distractor_failed",
        )
        assert r1.repairable is True, "distractor_failed code should be repairable"
        assert r1.failure_type == "distractor", f"expected 'distractor', got {r1.failure_type}"

        # failure_code='difficulty_misaligned' should classify as 'difficulty'
        r2 = svc.decide(
            failure_reason="generic error",
            repair_attempt_count=0,
            failure_code="difficulty_misaligned",
            target_theta=0.0,
            difficulty_signed_delta=1.0,
        )
        assert r2.repairable is True
        assert r2.failure_type == "difficulty", f"expected 'difficulty', got {r2.failure_type}"

        # Terminal code should not be repairable
        for code in ("evidence_insufficient", "grounding_failed", "duplicate_detected"):
            r3 = svc.decide(
                failure_reason="x",
                repair_attempt_count=0,
                failure_code=code,
            )
            assert r3.repairable is False, f"{code} should not be repairable"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# P1.5 — Judge rejection blocks validation
# ---------------------------------------------------------------------------

def test_validate_judge_rejection_blocks() -> None:
    """When LLM judge returns valid=false, the ValidationReport is invalid."""
    NAME = "F4. validate_judge_rejection_blocks"
    db = SessionLocal()
    try:
        from app.services.generated_question_validation_service import (
            GeneratedQuestionValidationService,
            ValidationReport,
        )

        # Mock the LLM to return a rejection
        mock_llm = MagicMock()
        mock_llm.is_available = True
        mock_llm.generate_chat.return_value = '{"valid": false, "feedback": "Ambiguous question", "ambiguity_found": true, "factual_error": false}'

        with patch(
            "app.services.generated_question_validation_service.LLMClient",
            return_value=mock_llm,
        ):
            svc = GeneratedQuestionValidationService(db)
            report = svc.validate(
                question_text="What is encryption?",
                options=[
                    {"text": "Protect confidentiality", "is_correct": True},
                    {"text": "Increase speed", "is_correct": False},
                    {"text": "Compress files", "is_correct": False},
                    {"text": "Generate randomness", "is_correct": False},
                ],
                explanation="Encryption converts plaintext to ciphertext.",
            )

        assert report.judge_ok is False, "judge should reject"
        assert report.judge_ambiguity is True
        assert report.valid is False, "judge rejection should make report invalid"
        assert any("judge" in i.lower() for i in report.issues)

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# P2.7 — Similarity normalization: None treated as 0.0
# ---------------------------------------------------------------------------

def test_confidence_none_similarity() -> None:
    """Confidence scoring handles chunks with similarity=None (treated as 0.0)."""
    NAME = "F5. confidence_none_similarity"
    try:
        chunks = [
            RetrievedChunk(
                chunk_id=1, document_id=0, course_name="test", title="A",
                text="encryption basics", chunk_index=0, topic_id=1, similarity=None,
            ),
            RetrievedChunk(
                chunk_id=2, document_id=0, course_name="test", title="B",
                text="encryption advanced", chunk_index=1, topic_id=1, similarity=None,
            ),
        ]
        v_report = ValidationReport(
            valid=True, issues=[], schema_ok=True,
            single_correct=True, non_duplicate=True, max_similarity=0.0,
        )
        gen = _fake_gen()
        service = QuestionConfidenceService()
        report = service.evaluate(
            retrieved_chunks=chunks, validation_report=v_report,
            retry_count=0, gen_output=gen,
        )
        # avg_sim is 0.0 (both None → 0.0), which is < 0.3 → penalty
        assert any("similarity" in r.lower() for r in report.reasons), (
            f"expected similarity-related reason, got {report.reasons}"
        )
        assert report.score < 100.0, "penalty should reduce score"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# P2.10 — Trace includes failure_code and judge fields
# ---------------------------------------------------------------------------

def test_trace_includes_failure_code_and_judge() -> None:
    """LangGraph trace includes failure_code, judge_participated, judge_ok."""
    NAME = "F6. trace_includes_failure_code_and_judge"
    try:
        from app.services.rag_telemetry_service import build_langgraph_trace

        # Case 1: No failure, judge participated and approved
        state_ok: dict = {
            "trace_id": uuid.uuid4().hex,
            "retry_count": 0,
            "retrieved_chunks": _fake_chunks(1),
            "confidence_report": ConfidenceReport(route="auto_approve", score=90.0, reasons=[]),
            "validation_report": ValidationReport(
                valid=True, schema_ok=True, single_correct=True,
                non_duplicate=True, judge_ok=True, judge_ambiguity=False,
                judge_factual_error=False, judge_feedback="Looks good",
            ),
            "repair_attempt_count": 0,
        }
        trace_ok = build_langgraph_trace(state_ok, _Settings())
        assert "failure_code" not in trace_ok, "success trace should have no failure_code"
        assert trace_ok["validation"]["judge_participated"] is True
        assert trace_ok["validation"]["judge_ok"] is True

        # Case 2: Graph aborted with failure_code, judge did not run
        state_fail: dict = {
            "trace_id": uuid.uuid4().hex,
            "retry_count": 0,
            "failure_code": "evidence_insufficient",
            "failure_reason": "insufficient evidence after repair",
            "repair_attempt_count": 1,
        }
        trace_fail = build_langgraph_trace(state_fail, _Settings())
        assert trace_fail["failure_code"] == "evidence_insufficient"
        assert trace_fail["failure_reason"] == "insufficient evidence after repair"

        # Case 3: Judge rejected (judge_ok=False)
        state_reject: dict = {
            "trace_id": uuid.uuid4().hex,
            "retry_count": 0,
            "retrieved_chunks": _fake_chunks(1),
            "validation_report": ValidationReport(
                valid=False, schema_ok=True, single_correct=True,
                non_duplicate=True, judge_ok=False, judge_ambiguity=True,
                judge_feedback="Ambiguous",
            ),
            "repair_attempt_count": 0,
        }
        trace_reject = build_langgraph_trace(state_reject, _Settings())
        assert trace_reject["validation"]["judge_participated"] is True
        assert trace_reject["validation"]["judge_ok"] is False

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# P1.6 — Dedup dedrift: dedup service is single source of truth
# ---------------------------------------------------------------------------

def test_validation_service_no_internal_dedup() -> None:
    """GeneratedQuestionValidationService.validate() does not call
    _check_duplicate — dedup is handled exclusively by QuestionDedupService."""
    NAME = "F7. validation_service_no_internal_dedup"
    try:
        from app.services.generated_question_validation_service import (
            GeneratedQuestionValidationService,
        )
        assert not hasattr(GeneratedQuestionValidationService, "_check_duplicate"), (
            "_check_duplicate should have been removed from validation service"
        )
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


# ---------------------------------------------------------------------------
# P0.1b — Streak reaches threshold after consecutive same-topic answers
# ---------------------------------------------------------------------------

def test_streak_reaches_threshold() -> None:
    """record_answer with before_order_index correctly counts same-topic streak."""
    NAME = "F8. streak_reaches_threshold"
    db = SessionLocal()
    try:
        from app.models import AdaptiveExamResponse, Question
        from app.models.enums import DifficultyLevel, CognitiveLevel
        from app.services.topic_streak_service import TopicStreakService

        ids = _setup(db)

        topic_a = Topic(phase_id=ids["phase_id"], name=f"__TopicA2_{uuid.uuid4().hex[:6]}__", description="tA2")
        db.add(topic_a)
        db.flush()
        sub_a = SubTopic(topic_id=topic_a.id, name=f"__SubA2__", description="sA2")
        db.add(sub_a)
        db.flush()

        ts = TopicStreakService(db, ids["student_id"])

        # Simulate 4 answers to topic A with increasing order_index
        # Each needs a unique question_id
        for oi in range(4):
            q = Question(
                text=f"P01b_q_{oi}_{uuid.uuid4().hex[:6]}", difficulty=DifficultyLevel.Medium,
                cognitive_level=CognitiveLevel.Application, question_type="SingleChoice",
                subtopic_id=sub_a.id, is_active=True,
            )
            db.add(q)
            db.flush()
            resp = AdaptiveExamResponse(
                adaptive_exam_id=ids["exam_id"], question_id=q.id,
                order_index=oi, is_correct=True, theta_before=0.0, theta_after=0.0,
            )
            db.add(resp)
            db.flush()
            info = ts.record_answer(
                ids["exam_id"], topic_a.id, 0.0, before_order_index=oi + 1,
            )

        # After 4 consecutive A answers, streak should be 4
        assert info.current_streak == 4, f"expected streak=4 after AAAA, got {info.current_streak}"
        assert info.threshold_reached is True, "threshold (4) should be reached"

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# Follow-up fix verification (F9–F13)
# ---------------------------------------------------------------------------


def test_validate_real_inputs_pass() -> None:
    """F9. When judge is unavailable, validate() returns valid=False (conservative)."""
    NAME = "F9. validate_real_inputs_blocks_when_judge_unavailable"
    db = SessionLocal()
    try:
        with patch("app.services.generated_question_validation_service.get_settings") as m_settings:
            m_settings.return_value = _Settings(
                RAG_ENABLED=True,
                RAG_REVIEW_REQUIRED=False,
                GENERATED_MCQ_OPTION_COUNT=4,
            )
            with patch("app.services.generated_question_validation_service.LLMClient") as m_llm_cls:
                m_llm = MagicMock()
                m_llm.is_available = False
                m_llm_cls.return_value = m_llm

                svc = GeneratedQuestionValidationService(db)
                report = svc.validate(
                    question_text="What is the primary purpose of encryption?",
                    options=[
                        {"text": "Protect confidentiality", "is_correct": True},
                        {"text": "Increase speed", "is_correct": False},
                        {"text": "Compress files", "is_correct": False},
                        {"text": "Generate randomness", "is_correct": False},
                    ],
                    explanation="Encryption converts plaintext to ciphertext for protection.",
                )

                assert report.valid is False, f"expected valid=False when judge unavailable, got {report.valid}"
                assert report.schema_ok is True
                assert report.single_correct is True
                assert report.non_duplicate is True
                assert report.judge_ok is False, "judge_ok should be False when unavailable"
                assert any("unavailable" in i.lower() or "judge" in i.lower() for i in report.issues), (
                    f"expected judge-related issue, got {report.issues}"
                )
                _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


def test_streak_savepoint_preserves_outer_transaction() -> None:
    """F10. record_answer preserves already-flushed outer state during duplicate
    progress recovery."""
    NAME = "F10. streak_savepoint_preserves_outer_tx"
    db = SessionLocal()
    try:
        from app.models import AdaptiveExam, AdaptiveExamResponse, Question
        from app.models.enums import CognitiveLevel, DifficultyLevel
        from app.services.topic_streak_service import TopicStreakService

        ids = _setup(db)

        topic_a = Topic(phase_id=ids["phase_id"], name=f"__TopicSA_{uuid.uuid4().hex[:6]}__", description="tSA")
        db.add(topic_a)
        db.flush()
        sub_a = SubTopic(topic_id=topic_a.id, name=f"__SubSA__", description="sSA")
        db.add(sub_a)
        db.flush()

        existing_progress = StudentTopicProgress(
            student_id=ids["student_id"],
            exam_id=ids["exam_id"],
            topic_id=topic_a.id,
            current_streak=2,
            questions_asked=2,
            generated_count=0,
            avg_theta=0.5,
        )
        db.add(existing_progress)
        db.flush()
        db.commit()

        stp_query_count = 0
        _real_query = db.query

        def _intercepting_query(*args, **kwargs):  # type: ignore[no-untyped-def]
            result = _real_query(*args, **kwargs)
            if args and args[0] is StudentTopicProgress:
                class _QProxy:
                    def __init__(self, q):
                        self._q = q
                    def filter(self, *a, **kw):
                        self._q = self._q.filter(*a, **kw)
                        return self
                    def first(self):
                        nonlocal stp_query_count
                        stp_query_count += 1
                        if stp_query_count == 1:
                            return None
                        return self._q.first()
                    def __getattr__(self, n):
                        return getattr(self._q, n)
                return _QProxy(result)
            return result

        q = Question(
            text=f"__RaceQ_{uuid.uuid4().hex[:6]}__",
            difficulty=DifficultyLevel.Medium,
            cognitive_level=CognitiveLevel.Application,
            question_type="SingleChoice",
            subtopic_id=sub_a.id,
            is_active=True,
        )
        db.add(q)
        db.flush()

        exam = db.get(AdaptiveExam, ids["exam_id"])
        assert exam is not None
        exam.answered_count = 1
        exam.current_theta = 0.35
        response = AdaptiveExamResponse(
            adaptive_exam_id=ids["exam_id"],
            question_id=q.id,
            order_index=1,
            is_correct=True,
            theta_before=0.0,
            theta_after=0.35,
        )
        db.add(response)
        db.flush()
        response_id = response.id

        ts = TopicStreakService(db, ids["student_id"])

        with patch.object(db, "query", new=_intercepting_query):
            info = ts.record_answer(
                ids["exam_id"], topic_a.id, 0.35, before_order_index=11,
            )

        assert stp_query_count >= 2, (
            f"expected ≥2 STP queries (race + recovery), got {stp_query_count}"
        )

        assert info.current_streak == 3, f"expected streak=3, got {info.current_streak}"
        assert info.topic_id == topic_a.id
        assert info.questions_asked == 3
        refreshed_exam = db.get(AdaptiveExam, ids["exam_id"])
        persisted_response = db.get(AdaptiveExamResponse, response_id)
        assert refreshed_exam is not None
        assert refreshed_exam.answered_count == 1, "outer exam update should survive duplicate recovery"
        assert abs(float(refreshed_exam.current_theta) - 0.35) < 1e-9
        assert persisted_response is not None, "flushed response should survive duplicate recovery"
        assert persisted_response.theta_after == 0.35

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


def test_retrieval_repair_clears_failure_code() -> None:
    """F11. retrieval_repair clears failure_code alongside failure_reason on success."""
    NAME = "F11. retrieval_repair_clears_failure_code"
    try:
        state: RAGGraphState = {
            "retry_count": 0,
            "streak_info": StreakInfo(
                topic_id=1, topic_name="Crypto", current_streak=4,
                questions_asked=4, generated_count=0, avg_theta=0.0,
                threshold_reached=True, can_generate=True,
                generation_attempted=False,
            ),
            "candidate_queries": ["old query"],
            "failure_reason": "Evidence insufficient: low_sim",
            "failure_code": "evidence_insufficient",
        }
        with patch("app.services.langgraph_rag_workflow.RetrievalRepairService") as m_cls:
            m_svc = MagicMock()
            m_svc.repair.return_value = ["broader query 1", "broader query 2"]
            m_cls.return_value = m_svc
            result = retrieval_repair(state)

        assert result.get("failure_reason") is None, f"failure_reason not cleared: {result.get('failure_reason')}"
        assert result.get("failure_code") is None, f"failure_code not cleared: {result.get('failure_code')}"
        assert result["retry_count"] == 1
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


def test_question_repair_clears_failure_code() -> None:
    """F12. question_repair clears failure_code alongside failure_reason on success."""
    NAME = "F12. question_repair_clears_failure_code"
    try:
        state: RAGGraphState = {
            "repair_attempt_count": 0,
            "repair_report": RepairReport(
                repairable=True,
                attempts_remaining=1,
                failure_type="difficulty_misaligned",
                hint={"target": "distractor_issue", "adjusted_theta": 0.0},
                issues=[],
            ),
            "failure_reason": "Difficulty misaligned: delta=0.5",
            "failure_code": "difficulty_misaligned",
        }
        result = question_repair(state)

        assert result.get("failure_reason") is None, f"failure_reason not cleared: {result.get('failure_reason')}"
        assert result.get("failure_code") is None, f"failure_code not cleared: {result.get('failure_code')}"
        assert result["repair_attempt_count"] == 1
        assert result["repair_hint"] == {"target": "distractor_issue", "adjusted_theta": 0.0}
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


def test_judge_results_persisted_in_validation_report() -> None:
    """F13. LangGraph persist node includes judge fields in validation_report JSON."""
    NAME = "F13. judge_results_persisted_in_vr_dict"
    try:
        v_report = ValidationReport(
            valid=True, issues=[], schema_ok=True, single_correct=True,
            non_duplicate=True, max_similarity=0.25,
            judge_ok=True, judge_feedback="Looks good", judge_ambiguity=False, judge_factual_error=False,
        )

        db = MagicMock()
        settings = _Settings(RAG_REVIEW_REQUIRED=False)

        state: RAGGraphState = {
            "topic_id": 1,
            "exam_id": 10,
            "gen_output": _fake_gen(),
            "validation_report": v_report,
            "retrieved_chunks": [],
            "confidence_report": ConfidenceReport(route="auto_approve", score=85.0, reasons=[]),
            "repair_attempt_count": 0,
        }

        with patch("app.models.rag.GeneratedQuestion") as MockGQ:
            mock_instance = MagicMock()
            mock_instance.id = 42
            MockGQ.return_value = mock_instance
            with patch("app.services.rag_telemetry_service.build_langgraph_trace", return_value={"trace_id": "t1"}):
                from app.services.langgraph_rag_workflow import _create_generated_question
                _create_generated_question(db, state, state["gen_output"], v_report, settings)

        call_kwargs = MockGQ.call_args
        vr_dict = call_kwargs.kwargs.get("validation_report") or call_kwargs[1].get("validation_report")

        assert vr_dict is not None, "validation_report not passed to GeneratedQuestion"
        assert vr_dict.get("judge_ok") is True, f"judge_ok missing or wrong: {vr_dict.get('judge_ok')}"
        assert vr_dict.get("judge_feedback") == "Looks good", f"judge_feedback wrong: {vr_dict.get('judge_feedback')}"
        assert vr_dict.get("judge_ambiguity") is False
        assert vr_dict.get("judge_factual_error") is False
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))


def test_judge_unavailable_blocks_validation() -> None:
    """F14. When LLM judge is unavailable (no API key), validate() returns valid=False."""
    NAME = "F14. judge_unavailable_blocks_validation"
    db = SessionLocal()
    try:
        with patch("app.services.generated_question_validation_service.get_settings") as m_settings:
            m_settings.return_value = _Settings(
                RAG_ENABLED=True,
                RAG_REVIEW_REQUIRED=False,
                GENERATED_MCQ_OPTION_COUNT=4,
            )
            with patch("app.services.generated_question_validation_service.LLMClient") as m_llm_cls:
                m_llm = MagicMock()
                m_llm.is_available = False
                m_llm_cls.return_value = m_llm

                svc = GeneratedQuestionValidationService(db)
                report = svc.validate(
                    question_text="What is the primary purpose of encryption?",
                    options=[
                        {"text": "Protect confidentiality", "is_correct": True},
                        {"text": "Increase speed", "is_correct": False},
                        {"text": "Compress files", "is_correct": False},
                        {"text": "Generate randomness", "is_correct": False},
                    ],
                    explanation="Encryption converts plaintext to ciphertext.",
                )

                assert report.valid is False, "judge unavailable should block validation"
                assert report.judge_ok is False, "judge_ok should be False when unavailable"
                assert any("unavailable" in i.lower() for i in report.issues), (
                    f"expected 'unavailable' in issues, got {report.issues}"
                )
                _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


def test_judge_exception_blocks_validation() -> None:
    """F15. When LLM judge call raises an exception, validate() returns valid=False."""
    NAME = "F15. judge_exception_blocks_validation"
    db = SessionLocal()
    try:
        with patch("app.services.generated_question_validation_service.get_settings") as m_settings:
            m_settings.return_value = _Settings(
                RAG_ENABLED=True,
                RAG_REVIEW_REQUIRED=True,
                GENERATED_MCQ_OPTION_COUNT=4,
            )
            with patch("app.services.generated_question_validation_service.LLMClient") as m_llm_cls:
                m_llm = MagicMock()
                m_llm.is_available = True
                m_llm.generate_chat.side_effect = RuntimeError("LLM service down")
                m_llm_cls.return_value = m_llm

                svc = GeneratedQuestionValidationService(db)
                report = svc.validate(
                    question_text="What is the primary purpose of encryption?",
                    options=[
                        {"text": "Protect confidentiality", "is_correct": True},
                        {"text": "Increase speed", "is_correct": False},
                        {"text": "Compress files", "is_correct": False},
                        {"text": "Generate randomness", "is_correct": False},
                    ],
                    explanation="Encryption converts plaintext to ciphertext.",
                )

                assert report.valid is False, "judge exception should block validation"
                assert report.judge_ok is False, "judge_ok should be False on exception"
                assert any("failed" in i.lower() for i in report.issues), (
                    f"expected 'failed' in issues, got {report.issues}"
                )
                _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


def test_outer_transaction_survives_get_progress_race() -> None:
    """F16. _get_progress duplicate recovery preserves already-flushed outer
    exam state."""
    NAME = "F16. outer_tx_survives_get_progress_race"
    db = SessionLocal()
    try:
        from app.models import AdaptiveExam, AdaptiveExamResponse, Question
        from app.models.enums import CognitiveLevel, DifficultyLevel
        from app.services.topic_streak_service import TopicStreakService

        ids = _setup(db)

        topic_b = Topic(phase_id=ids["phase_id"], name=f"__TopicOB_{uuid.uuid4().hex[:6]}__", description="tOB")
        db.add(topic_b)
        db.flush()

        existing = StudentTopicProgress(
            student_id=ids["student_id"],
            exam_id=ids["exam_id"],
            topic_id=topic_b.id,
            current_streak=0,
            questions_asked=0,
            generated_count=0,
            avg_theta=None,
        )
        db.add(existing)
        db.flush()
        db.commit()

        sub_b = SubTopic(topic_id=topic_b.id, name=f"__SubOB__", description="sOB")
        db.add(sub_b)
        db.flush()
        q = Question(
            text=f"__OuterRaceQ_{uuid.uuid4().hex[:6]}__",
            difficulty=DifficultyLevel.Medium,
            cognitive_level=CognitiveLevel.Application,
            question_type="SingleChoice",
            subtopic_id=sub_b.id,
            is_active=True,
        )
        db.add(q)
        db.flush()

        exam = db.get(AdaptiveExam, ids["exam_id"])
        assert exam is not None
        exam.answered_count = 1
        exam.current_theta = -0.2
        response = AdaptiveExamResponse(
            adaptive_exam_id=ids["exam_id"],
            question_id=q.id,
            order_index=1,
            is_correct=False,
            theta_before=0.0,
            theta_after=-0.2,
        )
        db.add(response)
        db.flush()
        response_id = response.id

        stp_query_count = 0
        _real_query = db.query

        def _intercepting_query(*args, **kwargs):  # type: ignore[no-untyped-def]
            result = _real_query(*args, **kwargs)
            if args and args[0] is StudentTopicProgress:
                class _QProxy:
                    def __init__(self, q):
                        self._q = q
                    def filter(self, *a, **kw):
                        self._q = self._q.filter(*a, **kw)
                        return self
                    def first(self):
                        nonlocal stp_query_count
                        stp_query_count += 1
                        if stp_query_count == 1:
                            return None
                        return self._q.first()
                    def __getattr__(self, n):
                        return getattr(self._q, n)
                return _QProxy(result)
            return result

        ts = TopicStreakService(db, ids["student_id"])

        with patch.object(db, "query", new=_intercepting_query):
            progress = ts._get_progress(ids["exam_id"], topic_b.id)

        assert stp_query_count >= 2, (
            f"expected ≥2 STP queries (race + recovery), got {stp_query_count}"
        )

        assert progress is not None
        assert progress.student_id == ids["student_id"]
        assert progress.topic_id == topic_b.id
        assert progress.current_streak == 0
        assert progress.avg_theta is None
        refreshed_exam = db.get(AdaptiveExam, ids["exam_id"])
        persisted_response = db.get(AdaptiveExamResponse, response_id)
        assert refreshed_exam is not None
        assert refreshed_exam.answered_count == 1, "outer exam update should survive duplicate recovery"
        assert abs(float(refreshed_exam.current_theta) - (-0.2)) < 1e-9
        assert persisted_response is not None, "flushed response should survive duplicate recovery"
        assert persisted_response.theta_after == -0.2

        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# Topic-lock orchestration tests
# ---------------------------------------------------------------------------

def _make_question_with_choices(db, subtopic_id: int, tag: str) -> tuple[Question, Choice]:
    """Create a question with one correct choice. Returns (question, correct_choice)."""
    from app.models.enums import DifficultyLevel, CognitiveLevel

    q = Question(
        text=f"Q_{tag}_{uuid.uuid4().hex[:8]}",
        difficulty=DifficultyLevel.Medium,
        cognitive_level=CognitiveLevel.Application,
        question_type="SingleChoice",
        subtopic_id=subtopic_id,
        is_active=True,
    )
    db.add(q)
    db.flush()
    c = Choice(question_id=q.id, text=f"Correct_{tag}", is_correct=True)
    db.add(c)
    db.flush()
    return q, c


def _make_exam_with_two_topics(db, student_id: int, phase_id: int) -> dict:
    """Create exam + two topics with 5 questions each. Returns IDs dict."""
    existing_phase = db.query(Phase).filter(Phase.id == phase_id).first()
    if not existing_phase:
        db.add(Phase(id=phase_id, name=f"_TL_Phase_{phase_id}", description="tl-test-phase"))
        db.flush()

    existing_user = db.query(User).filter(User.id == student_id).first()
    if not existing_user:
        db.add(User(id=student_id, full_name=f"_TL_User_{student_id}", email=f"_tl_{student_id}@test.local", hashed_password="x", role=UserRole.Student))
        db.flush()

    topic_a = Topic(phase_id=phase_id, name=f"_TL_A_{uuid.uuid4().hex[:6]}", description="tlA")
    db.add(topic_a)
    db.flush()
    sub_a = SubTopic(topic_id=topic_a.id, name=f"_TL_SubA_{uuid.uuid4().hex[:6]}", description="sA")
    db.add(sub_a)
    db.flush()

    topic_b = Topic(phase_id=phase_id, name=f"_TL_B_{uuid.uuid4().hex[:6]}", description="tlB")
    db.add(topic_b)
    db.flush()
    sub_b = SubTopic(topic_id=topic_b.id, name=f"_TL_SubB_{uuid.uuid4().hex[:6]}", description="sB")
    db.add(sub_b)
    db.flush()

    exam = AdaptiveExam(
        student_id=student_id, phase_id=phase_id,
        status=ExamStatus.InProgress, max_questions=20,
        answered_count=0, current_theta=0.0,
    )
    db.add(exam)
    db.flush()

    a_qs, a_cs, b_qs, b_cs = [], [], [], []
    for i in range(5):
        q, c = _make_question_with_choices(db, sub_a.id, f"tlA{i}")
        a_qs.append(q)
        a_cs.append(c)
    for i in range(5):
        q, c = _make_question_with_choices(db, sub_b.id, f"tlB{i}")
        b_qs.append(q)
        b_cs.append(c)

    return {
        "student_id": student_id,
        "phase_id": phase_id,
        "exam_id": exam.id,
        "topic_a_id": topic_a.id,
        "topic_b_id": topic_b.id,
        "a_questions": a_qs,
        "a_choices": a_cs,
        "b_questions": b_qs,
        "b_choices": b_cs,
    }


def test_topic_lock_basic_flow() -> None:
    """TL1: 3 regular As → 4th triggers generation → generated question returned."""
    NAME = "TL1. topic_lock_basic_flow"
    db = SessionLocal()
    try:
        ids = _make_exam_with_two_topics(db, 99001, 99001)
        settings = _Settings(RAG_ENABLED=True, RAG_LANGGRAPH_ENABLED=False)

        for i in range(3):
            resp = submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                ids["a_questions"][i].id, ids["a_choices"][i].id,
            )
            q = resp["current_question"]
            assert q["topic_id"] == ids["topic_a_id"], f"step{i+1}: expected topic A"
            assert q["is_generated"] is False

        exam = db.query(AdaptiveExam).filter(AdaptiveExam.id == ids["exam_id"]).first()
        ts = __import__("app.services.topic_streak_service", fromlist=["TopicStreakService"]).TopicStreakService(db, ids["student_id"])
        si = ts.get_streak(ids["exam_id"], ids["topic_a_id"])
        assert not si.threshold_reached, "streak should not yet reach threshold"

        fake_gq = _make_fake_gq(db, ids["exam_id"], ids["topic_a_id"])

        with patch("app.services.adaptive_exam_service.get_settings", return_value=settings), \
             patch("app.services.topic_streak_service.get_settings", return_value=settings), \
             patch("app.services.adaptive_exam_service._try_generate_next", return_value=fake_gq):
            resp = submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                ids["a_questions"][3].id, ids["a_choices"][3].id,
            )
        q = resp["current_question"]
        assert q["is_generated"] is True, "expected generated question at threshold"
        assert q["topic_id"] == ids["topic_a_id"]
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


def test_topic_lock_blocks_jump() -> None:
    """TL2: During streak-building, IRT cannot jump to topic B."""
    NAME = "TL2. topic_lock_blocks_jump"
    db = SessionLocal()
    try:
        ids = _make_exam_with_two_topics(db, 99002, 99002)

        resp = submit_adaptive_answer(
            db, ids["student_id"], ids["exam_id"],
            ids["a_questions"][0].id, ids["a_choices"][0].id,
        )
        exam = db.query(AdaptiveExam).filter(AdaptiveExam.id == ids["exam_id"]).first()
        assert exam.locked_topic_id == ids["topic_a_id"], "should lock to topic A"

        resp2 = submit_adaptive_answer(
            db, ids["student_id"], ids["exam_id"],
            ids["a_questions"][1].id, ids["a_choices"][1].id,
        )
        q2 = resp2["current_question"]
        assert q2["topic_id"] == ids["topic_a_id"], "should still be on topic A"
        assert q2["is_generated"] is False

        exam = db.query(AdaptiveExam).filter(AdaptiveExam.id == ids["exam_id"]).first()
        assert exam.locked_topic_id == ids["topic_a_id"], "lock should persist"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


def test_non_consecutive_no_trigger() -> None:
    """TL3: A B A does NOT trigger generation for A (streak reset on topic change)."""
    NAME = "TL3. non_consecutive_no_trigger"
    db = SessionLocal()
    try:
        ids = _make_exam_with_two_topics(db, 99003, 99003)

        submit_adaptive_answer(
            db, ids["student_id"], ids["exam_id"],
            ids["a_questions"][0].id, ids["a_choices"][0].id,
        )
        submit_adaptive_answer(
            db, ids["student_id"], ids["exam_id"],
            ids["b_questions"][0].id, ids["b_choices"][0].id,
        )
        ts = __import__("app.services.topic_streak_service", fromlist=["TopicStreakService"]).TopicStreakService(db, ids["student_id"])
        si_a = ts.get_streak(ids["exam_id"], ids["topic_a_id"])
        assert si_a.current_streak == 0, "A streak should reset after B answer"
        assert not si_a.threshold_reached

        settings = _Settings(RAG_ENABLED=True, RAG_LANGGRAPH_ENABLED=False)
        with patch("app.services.adaptive_exam_service.get_settings", return_value=settings), \
             patch("app.services.topic_streak_service.get_settings", return_value=settings):
            resp = submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                ids["a_questions"][1].id, ids["a_choices"][1].id,
            )
        q = resp["current_question"]
        assert q["is_generated"] is False, "should NOT trigger generation"
        assert q["topic_id"] == ids["topic_a_id"]
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


def test_generated_does_not_count_streak() -> None:
    """TL4: Answering a generated question does NOT increment regular streak."""
    NAME = "TL4. generated_does_not_count_streak"
    db = SessionLocal()
    try:
        ids = _make_exam_with_two_topics(db, 99004, 99004)
        settings = _Settings(RAG_ENABLED=True, RAG_LANGGRAPH_ENABLED=False)

        for i in range(3):
            submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                ids["a_questions"][i].id, ids["a_choices"][i].id,
            )

        fake_gq = _make_fake_gq(db, ids["exam_id"], ids["topic_a_id"])

        def _fake_gen_inc(db_, exam_id_, topic_id_, theta_, student_id_):
            from app.services.topic_streak_service import TopicStreakService as _TSS
            _TSS(db_, student_id_).increment_generated(exam_id_, topic_id_)
            return fake_gq

        with patch("app.services.adaptive_exam_service.get_settings", return_value=settings), \
             patch("app.services.topic_streak_service.get_settings", return_value=settings), \
             patch("app.services.adaptive_exam_service._try_generate_next", side_effect=_fake_gen_inc):
            resp = submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                ids["a_questions"][3].id, ids["a_choices"][3].id,
            )
        gq_id = resp["current_question"]["id"]

        with patch("app.services.adaptive_exam_service.get_settings", return_value=settings), \
             patch("app.services.topic_streak_service.get_settings", return_value=settings):
            submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                gq_id, 1,
            )

        ts = __import__("app.services.topic_streak_service", fromlist=["TopicStreakService"]).TopicStreakService(db, ids["student_id"])
        si = ts.get_streak(ids["exam_id"], ids["topic_a_id"])
        assert si.generated_count == 1, "generated_count should be 1"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


def test_generation_failure_fallback() -> None:
    """TL5: Failed generation → one more regular Q from same topic."""
    NAME = "TL5. generation_failure_fallback"
    db = SessionLocal()
    try:
        ids = _make_exam_with_two_topics(db, 99005, 99005)
        settings = _Settings(RAG_ENABLED=True, RAG_LANGGRAPH_ENABLED=False)

        # Create one extra topic-A question so fallback has one left after 4 consumed
        extra_q, extra_c = _make_question_with_choices(db, db.query(SubTopic).filter(SubTopic.topic_id == ids["topic_a_id"]).first().id, "tlA_extra")
        db.flush()

        for i in range(3):
            submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                ids["a_questions"][i].id, ids["a_choices"][i].id,
            )

        with patch("app.services.adaptive_exam_service.get_settings", return_value=settings), \
             patch("app.services.topic_streak_service.get_settings", return_value=settings), \
             patch("app.services.adaptive_exam_service._try_generate_next", return_value=None):
            resp = submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                ids["a_questions"][3].id, ids["a_choices"][3].id,
            )

        q = resp["current_question"]
        assert q["is_generated"] is False, "generation failed → regular Q"
        assert q["topic_id"] == ids["topic_a_id"], "should stay on topic A"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


def test_no_infinite_retry_loop() -> None:
    """TL6: After failed generation, subsequent answers do NOT retry generation."""
    NAME = "TL6. no_infinite_retry_loop"
    db = SessionLocal()
    try:
        ids = _make_exam_with_two_topics(db, 99006, 99006)
        settings = _Settings(RAG_ENABLED=True, RAG_LANGGRAPH_ENABLED=False)

        for i in range(4):
            submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                ids["a_questions"][i].id, ids["a_choices"][i].id,
            )

        with patch("app.services.adaptive_exam_service.get_settings", return_value=settings), \
             patch("app.services.topic_streak_service.get_settings", return_value=settings), \
             patch("app.services.adaptive_exam_service.RAGRetrievalService") as mock_ret:
            mock_ret.return_value.retrieve.return_value = []
            mock_ret.return_value.close.return_value = None
            resp5 = submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                ids["a_questions"][4].id, ids["a_choices"][4].id,
            )

        ts = __import__("app.services.topic_streak_service", fromlist=["TopicStreakService"]).TopicStreakService(db, ids["student_id"])
        si = ts.get_streak(ids["exam_id"], ids["topic_a_id"])
        assert si.generation_attempted, "generation_attempted should be True"

        # If more questions exist on topic A, answer them — should NOT retry generation
        if len(ids["a_questions"]) > 5:
            with patch("app.services.adaptive_exam_service.get_settings", return_value=settings), \
                 patch("app.services.topic_streak_service.get_settings", return_value=settings):
                resp6 = submit_adaptive_answer(
                    db, ids["student_id"], ids["exam_id"],
                    ids["a_questions"][5].id, ids["a_choices"][5].id,
                )
            q6 = resp6["current_question"]
            assert q6["is_generated"] is False, "should NOT retry generation"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


def test_move_to_topic_b_after_generated() -> None:
    """TL7: After answering generated Q for A, next Q moves away from A (not generated)."""
    NAME = "TL7. move_to_topic_b_after_generated"
    db = SessionLocal()
    try:
        ids = _make_exam_with_two_topics(db, 99007, 99007)
        settings = _Settings(RAG_ENABLED=True, RAG_LANGGRAPH_ENABLED=False)

        for i in range(3):
            submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                ids["a_questions"][i].id, ids["a_choices"][i].id,
            )

        fake_gq = _make_fake_gq(db, ids["exam_id"], ids["topic_a_id"])

        with patch("app.services.adaptive_exam_service.get_settings", return_value=settings), \
             patch("app.services.topic_streak_service.get_settings", return_value=settings), \
             patch("app.services.adaptive_exam_service._try_generate_next", return_value=fake_gq):
            resp_gen = submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                ids["a_questions"][3].id, ids["a_choices"][3].id,
            )
        gq_id = resp_gen["current_question"]["id"]

        with patch("app.services.adaptive_exam_service.get_settings", return_value=settings), \
             patch("app.services.topic_streak_service.get_settings", return_value=settings):
            resp_next = submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                gq_id, 1,
            )

        q_next = resp_next["current_question"]
        assert q_next is not None, "should have a next question"
        assert q_next["is_generated"] is False, "next should be regular Q"
        assert q_next["topic_id"] != ids["topic_a_id"], "must move away from consumed topic A"

        exam = db.query(AdaptiveExam).filter(AdaptiveExam.id == ids["exam_id"]).first()
        assert exam.locked_topic_id is None, "lock should be cleared"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


def test_langgraph_topic_lock_flow() -> None:
    """TL8: Same topic-lock flow works with RAG_LANGGRAPH_ENABLED=true."""
    NAME = "TL8. langgraph_topic_lock_flow"
    db = SessionLocal()
    try:
        ids = _make_exam_with_two_topics(db, 99008, 99008)
        settings = _Settings(RAG_ENABLED=True, RAG_LANGGRAPH_ENABLED=True)

        for i in range(3):
            resp = submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                ids["a_questions"][i].id, ids["a_choices"][i].id,
            )
            q = resp["current_question"]
            assert q["topic_id"] == ids["topic_a_id"]
            assert q["is_generated"] is False

        fake_gq = _make_fake_gq(db, ids["exam_id"], ids["topic_a_id"])

        with patch("app.services.adaptive_exam_service.get_settings", return_value=settings), \
             patch("app.services.topic_streak_service.get_settings", return_value=settings), \
             patch("app.services.adaptive_exam_service._try_generate_next", return_value=fake_gq):
            resp = submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                ids["a_questions"][3].id, ids["a_choices"][3].id,
            )

        q = resp["current_question"]
        assert q["is_generated"] is True, "LangGraph should return generated question"
        assert q["topic_id"] == ids["topic_a_id"]
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# Issue-fix tests: max_questions cap + pending generated question recovery
# ---------------------------------------------------------------------------

def test_max_questions_cap_preserved() -> None:
    """IF1: Threshold reached on final allowed regular question → exam completes, no generated Q."""
    NAME = "IF1. max_questions_cap_preserved"
    db = SessionLocal()
    try:
        ids = _make_exam_with_two_topics(db, 98001, 98001)
        # Set max_questions=4 so the 4th answer triggers completion
        exam = db.query(AdaptiveExam).filter(AdaptiveExam.id == ids["exam_id"]).first()
        exam.max_questions = 4
        db.flush()

        settings = _Settings(RAG_ENABLED=True, RAG_LANGGRAPH_ENABLED=False)
        for i in range(4):
            with patch("app.services.adaptive_exam_service.get_settings", return_value=settings), \
                 patch("app.services.topic_streak_service.get_settings", return_value=settings):
                resp = submit_adaptive_answer(
                    db, ids["student_id"], ids["exam_id"],
                    ids["a_questions"][i].id, ids["a_choices"][i].id,
                )

        # On the 4th answer the exam should complete
        assert resp["status"].value == "Completed", f"expected Completed, got {resp['status']}"
        assert resp["result"] is not None, "expected result dict"
        assert resp["answered_count"] == 4
        assert resp["current_question"] is None, "no next question after completion"

        exam = db.query(AdaptiveExam).filter(AdaptiveExam.id == ids["exam_id"]).first()
        assert exam.pending_generated_question_id is None, "pending should be cleared on completion"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


def test_pending_generated_question_survives_refresh() -> None:
    """IF2: After generation succeeds, get_adaptive_exam returns the same generated Q."""
    NAME = "IF2. pending_generated_question_survives_refresh"
    db = SessionLocal()
    try:
        ids = _make_exam_with_two_topics(db, 98002, 98002)
        settings = _Settings(RAG_ENABLED=True, RAG_LANGGRAPH_ENABLED=False)

        for i in range(3):
            submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                ids["a_questions"][i].id, ids["a_choices"][i].id,
            )

        fake_gq = _make_fake_gq(db, ids["exam_id"], ids["topic_a_id"])

        with patch("app.services.adaptive_exam_service.get_settings", return_value=settings), \
             patch("app.services.topic_streak_service.get_settings", return_value=settings), \
             patch("app.services.adaptive_exam_service._try_generate_next", return_value=fake_gq):
            resp = submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                ids["a_questions"][3].id, ids["a_choices"][3].id,
            )

        gq_id = resp["current_question"]["id"]
        assert resp["current_question"]["is_generated"] is True

        with patch("app.services.topic_streak_service.get_settings", return_value=settings):
            refresh_resp = get_adaptive_exam(db, ids["student_id"], ids["exam_id"])

        rq = refresh_resp["current_question"]
        assert rq is not None, "refresh should return a question"
        assert rq["id"] == gq_id, f"expected same generated Q {gq_id}, got {rq['id']}"
        assert rq["is_generated"] is True, "refresh should return generated Q"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


def test_pending_generated_question_clears_after_answer() -> None:
    """IF3: After answering the pending generated Q, pending state clears and topic moves on."""
    NAME = "IF3. pending_generated_question_clears_after_answer"
    db = SessionLocal()
    try:
        ids = _make_exam_with_two_topics(db, 98003, 98003)
        settings = _Settings(RAG_ENABLED=True, RAG_LANGGRAPH_ENABLED=False)

        for i in range(3):
            submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                ids["a_questions"][i].id, ids["a_choices"][i].id,
            )

        fake_gq = _make_fake_gq(db, ids["exam_id"], ids["topic_a_id"])

        with patch("app.services.adaptive_exam_service.get_settings", return_value=settings), \
             patch("app.services.topic_streak_service.get_settings", return_value=settings), \
             patch("app.services.adaptive_exam_service._try_generate_next", return_value=fake_gq):
            resp_gen = submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                ids["a_questions"][3].id, ids["a_choices"][3].id,
            )
        gq_id = resp_gen["current_question"]["id"]

        exam = db.query(AdaptiveExam).filter(AdaptiveExam.id == ids["exam_id"]).first()
        assert exam.pending_generated_question_id == gq_id

        with patch("app.services.adaptive_exam_service.get_settings", return_value=settings), \
             patch("app.services.topic_streak_service.get_settings", return_value=settings):
            resp_next = submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                gq_id, 1,
            )

        exam = db.query(AdaptiveExam).filter(AdaptiveExam.id == ids["exam_id"]).first()
        assert exam.pending_generated_question_id is None, "pending should be cleared after answer"

        q_next = resp_next["current_question"]
        assert q_next is not None, "should have a next question"
        assert q_next["is_generated"] is False
        assert q_next["topic_id"] != ids["topic_a_id"], "must move away from consumed topic A"

        with patch("app.services.topic_streak_service.get_settings", return_value=settings):
            refresh_resp = get_adaptive_exam(db, ids["student_id"], ids["exam_id"])
        rq = refresh_resp["current_question"]
        assert rq is not None
        assert rq["id"] != gq_id, "should not return the old generated Q"
        assert rq["is_generated"] is False
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


def test_no_duplicate_generated_issuance_on_refresh() -> None:
    """IF4: Repeated GET before answering the generated Q returns the same one, no new generation."""
    NAME = "IF4. no_duplicate_generated_issuance_on_refresh"
    db = SessionLocal()
    try:
        ids = _make_exam_with_two_topics(db, 98004, 98004)
        settings = _Settings(RAG_ENABLED=True, RAG_LANGGRAPH_ENABLED=False)

        for i in range(3):
            submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                ids["a_questions"][i].id, ids["a_choices"][i].id,
            )

        fake_gq = _make_fake_gq(db, ids["exam_id"], ids["topic_a_id"])

        with patch("app.services.adaptive_exam_service.get_settings", return_value=settings), \
             patch("app.services.topic_streak_service.get_settings", return_value=settings), \
             patch("app.services.adaptive_exam_service._try_generate_next", return_value=fake_gq):
            resp = submit_adaptive_answer(
                db, ids["student_id"], ids["exam_id"],
                ids["a_questions"][3].id, ids["a_choices"][3].id,
            )
        gq_id = resp["current_question"]["id"]

        for _ in range(3):
            with patch("app.services.topic_streak_service.get_settings", return_value=settings):
                refresh_resp = get_adaptive_exam(db, ids["student_id"], ids["exam_id"])
            rq = refresh_resp["current_question"]
            assert rq is not None
            assert rq["id"] == gq_id, f"expected same generated Q {gq_id}, got {rq['id']}"
            assert rq["is_generated"] is True

        gq_count = (
            db.query(GeneratedQuestion)
            .filter(
                GeneratedQuestion.source_exam_id == ids["exam_id"],
                GeneratedQuestion.topic_id == ids["topic_a_id"],
            )
            .count()
        )
        assert gq_count == 1, f"expected exactly 1 generated Q, got {gq_count}"
        _ok(NAME)
    except Exception as exc:
        _fail(NAME, str(exc))
    finally:
        db.rollback()
        db.close()


_ALL_TESTS = [
    # Phase 1
    test_legacy_success,
    test_langgraph_success,
    test_streak_gate_blocks,
    test_empty_retrieval,
    test_validation_failure,
    test_persists_question_and_evidence,
    test_increments_generated_count,
    test_equivalence,
    # Phase 2.1
    test_query_planner_creates_multiple_queries,
    test_query_planner_deterministic,
    test_query_planner_difficulty_bands,
    test_langgraph_multi_query_routing,
    test_langgraph_single_query_fallback,
    test_langgraph_empty_planner_fallback,
    test_langgraph_planner_exception_fallback,
    test_legacy_path_never_uses_planner,
    test_phase1_equivalence_with_planner,
    # Phase 2.2
    test_reranker_node_invoked_in_langgraph,
    test_reranker_receives_correct_queries,
    test_reranker_exception_fallback,
    test_lexical_overlap_deterministic,
    test_legacy_path_bypasses_reranker,
    test_reranked_order_passed_to_generate,
    test_reranker_sorts_highest_score_first,
    test_reranker_tie_breaks_by_original_position,
    # Phase 2.3
    test_compressor_node_invoked_in_langgraph,
    test_compressed_chunks_passed_to_generate,
    test_evidence_gate_blocks_generation,
    test_evidence_gate_no_persistence,
    test_compressor_exception_fallback,
    test_evidence_gate_exception_blocks,
    test_legacy_path_bypasses_compressor_and_gate,
    test_compression_deduplication,
    test_evidence_validation_insufficient_chunks,
    test_evidence_validation_low_similarity,
    test_evidence_validation_no_high_quality,
    test_evidence_validation_sufficient,
    # Phase 2.4
    test_retrieval_repair_service_generates_broader_queries,
    test_retrieval_repair_service_deduplicates_against_original,
    test_retrieval_repair_service_fallback_when_all_tried,
    test_retrieval_repair_triggered_after_insufficient_evidence,
    test_success_after_repair_persists,
    test_double_failure_aborts_safely,
    test_legacy_path_unchanged_by_repair,
    test_repair_exception_fallback,
    # Phase 2.5
    test_duplicate_gate_invoked_in_langgraph,
    test_duplicate_blocks_persistence,
    test_non_duplicate_persists,
    test_duplicate_generated_question_blocked,
    test_no_persist_on_duplicate,
    test_legacy_path_unchanged_by_duplicate_gate,
    # Phase 2.6
    test_confidence_gate_invoked_in_langgraph,
    test_high_confidence_auto_approve,
    test_medium_confidence_human_review,
    test_low_confidence_reject,
    test_global_review_overrides_auto_approve,
    test_confidence_scoring_deterministic,
    test_legacy_path_unchanged_by_confidence_gate,
    test_confidence_exception_fallback_persists_draft,
    # Phase 2.7
    test_langgraph_trace_in_validation_report,
    test_telemetry_exception_does_not_block_persist,
    # Phase 2.8
    test_grounding_well_supported,
    test_grounding_unsupported_answer,
    test_grounding_unsupported_explanation,
    test_grounding_no_chunks,
    test_grounding_validator_invoked_in_langgraph,
    test_grounding_failure_blocks_persistence,
    test_grounding_exception_blocks_persist,
    test_grounding_report_in_validation_report,
    test_legacy_path_unchanged_by_grounding,
    # Phase 2.8b — grounding fixes
    test_grounding_unsupported_stem_blocks,
    test_grounding_empty_explanation_blocks,
    test_grounding_empty_answer_blocks,
    test_grounding_empty_question_stem_blocks,
    # Phase 2.9 — distractor validator
    test_distractor_valid_passes,
    test_distractor_duplicates_block,
    test_distractor_too_similar_to_answer_blocks,
    test_distractor_empty_blocks,
    test_distractor_too_few_blocks,
    test_distractor_validator_invoked_in_langgraph,
    test_distractor_failure_blocks_persistence,
    test_distractor_report_in_validation_report,
    test_legacy_path_unchanged_by_distractor,
    # Phase 2.9b — distractor hardening
    test_distractor_empty_correct_answer_blocks,
    test_distractor_stopword_only_blocks,
    test_distractor_empty_correct_answer_blocks_persist,
    # Phase 2.10 — difficulty calibration
    test_difficulty_aligned_passes,
    test_difficulty_too_easy_blocks,
    test_difficulty_too_hard_blocks,
    test_difficulty_missing_estimate_blocks,
    test_difficulty_calibrator_invoked_in_langgraph,
    test_difficulty_misalignment_blocks_persistence,
    test_difficulty_exception_blocks_persist,
    test_difficulty_report_in_validation_report,
    test_legacy_path_unchanged_by_difficulty,
    test_difficulty_missing_estimate_blocks_node,
    # Phase 2.11 — question repair / regeneration loop
    test_repair_decision_classifies_distractor,
    test_repair_decision_classifies_difficulty,
    test_repair_decision_blocks_evidence,
    test_repair_decision_blocks_grounding,
    test_repair_decision_blocks_duplicate,
    test_repair_budget_enforced,
    test_repair_hint_adjusts_theta_too_hard,
    test_repair_hint_adjusts_theta_too_easy,
    test_repair_triggered_on_distractor_failure,
    test_second_failure_after_repair_aborts,
    test_non_repairable_failure_skips_repair,
    test_max_repair_count_enforced,
    test_repair_metadata_persisted,
    # Phase 2.11b — repair direction + missing difficulty hardening
    test_missing_difficulty_blocks_langgraph_path,
    test_real_json_parsing_missing_difficulty_blocks,
    # Phase 2.12 — artifact validation (citations + distractor rationale)
    test_artifact_valid_citations_and_rationale_persists,
    test_artifact_missing_citations_blocks,
    test_artifact_empty_citations_blocks,
    test_artifact_short_citations_blocks,
    test_artifact_fabricated_citations_blocks,
    test_artifact_missing_rationale_blocks,
    test_artifact_incomplete_rationale_blocks,
    test_artifact_rationale_with_correct_entry_blocks,
    test_artifact_empty_rationale_text_blocks,
    test_artifact_config_off_citations_skipped,
    test_artifact_config_off_rationale_skipped,
    test_artifact_both_config_off,
    test_legacy_path_unchanged_by_artifact,
    test_artifact_report_in_validation_report,
    test_artifact_failure_blocks_persistence,
    test_artifact_validator_invoked_in_langgraph,
    test_artifact_exception_blocks_persist,
    test_artifact_partial_invalid_citations_blocks,
    test_generation_preserves_non_string_rationale_for_validation,
    # Fix-plan verification
    test_streak_before_order_index,
    test_difficulty_map_logit_scale,
    test_repair_decision_uses_failure_code,
    test_validate_judge_rejection_blocks,
    test_confidence_none_similarity,
    test_trace_includes_failure_code_and_judge,
    test_validation_service_no_internal_dedup,
    test_streak_reaches_threshold,
    # Topic-lock orchestration
    test_topic_lock_basic_flow,
    test_topic_lock_blocks_jump,
    test_non_consecutive_no_trigger,
    test_generated_does_not_count_streak,
    test_generation_failure_fallback,
    test_no_infinite_retry_loop,
    test_move_to_topic_b_after_generated,
    test_langgraph_topic_lock_flow,
    # Issue fixes — max_questions cap + pending generated question recovery
    test_max_questions_cap_preserved,
    test_pending_generated_question_survives_refresh,
    test_pending_generated_question_clears_after_answer,
    test_no_duplicate_generated_issuance_on_refresh,
    # Follow-up fixes
    test_validate_real_inputs_pass,
    test_streak_savepoint_preserves_outer_transaction,
    test_retrieval_repair_clears_failure_code,
    test_question_repair_clears_failure_code,
    test_judge_results_persisted_in_validation_report,
    test_judge_unavailable_blocks_validation,
    test_judge_exception_blocks_validation,
    test_outer_transaction_survives_get_progress_race,
]


def main() -> int:
    print("=" * 60)
    print("LangGraph Phase 1 + 2.1–2.12 + Fix-Plan + Follow-up Verification")
    print("=" * 60)

    for fn in _ALL_TESTS:
        fn()

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    failed = sum(1 for _, ok, _ in _RESULTS if not ok)

    print()
    for name, ok, msg in _RESULTS:
        tag = "PASS" if ok else "FAIL"
        line = f"  [{tag}] {name}"
        if msg:
            line += f"  --  {msg}"
        print(line)

    print()
    print("-" * 60)
    print(f"  {passed} passed, {failed} failed, {len(_RESULTS)} total")
    print("-" * 60)

    if failed:
        print("\nSOME TESTS FAILED")
        return 1

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

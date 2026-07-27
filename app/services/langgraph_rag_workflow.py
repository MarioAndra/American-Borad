from __future__ import annotations

from uuid import uuid4

from langgraph.graph import END, StateGraph

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.enums import GeneratedQuestionStatus
from app.services.generated_question_service import GenerationInput, GeneratedQuestionService
from app.services.generated_question_validation_service import GeneratedQuestionValidationService
from app.services.generated_artifact_validation_service import GeneratedArtifactValidationService
from app.services.langgraph_rag_state import RAGGraphState
from app.services.query_planning_service import QueryPlannerService
from app.services.rag_retrieval_service import RAGRetrievalService
from app.services.context_compression_service import ContextCompressionService
from app.services.difficulty_calibration_service import DifficultyCalibrationService
from app.services.distractor_validation_service import DistractorValidationService
from app.services.evidence_validation_service import EvidenceValidationService
from app.services.retrieval_rerank_service import RerankerService
from app.services.retrieval_repair_service import RetrievalRepairService
from app.services.grounding_validation_service import GroundingValidationService
from app.services.question_dedup_service import QuestionDedupService
from app.services.question_confidence_service import (
    ConfidenceReport,
    QuestionConfidenceService,
)
from app.services.question_repair_service import QuestionRepairService
from app.services.topic_streak_service import TopicStreakService

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Node implementations — thin adapters around existing services
# ---------------------------------------------------------------------------


def gate_check(state: RAGGraphState) -> dict:
    """Check topic-streak eligibility for question generation."""
    settings = get_settings()
    if not settings.RAG_ENABLED:
        return {
            "failure_reason": "RAG_ENABLED is false",
            "failure_code": "rag_disabled",
        }

    db = state["db"]
    streak_service = TopicStreakService(db, state["student_id"])
    streak_info = streak_service.get_streak(state["exam_id"], state["topic_id"])

    if not streak_info.can_generate:
        return {
            "failure_reason": "Streak gate not met",
            "failure_code": "streak_not_met",
        }

    return {"streak_info": streak_info}


def query_planner(state: RAGGraphState) -> dict:
    """Generate candidate retrieval queries from topic + theta context.

    Best-effort: if the planner raises or returns unusable output,
    fall back to an empty list so the retrieve node uses the legacy
    single-query path (``streak_info.topic_name``).
    """
    try:
        streak_info = state["streak_info"]
        planner = QueryPlannerService()
        plan = planner.plan(
            topic_name=streak_info.topic_name,
            theta=state["theta"],
        )
        queries = [q for q in plan.queries if q and q.strip()]
        log.info("QueryPlanner: %d queries planned for topic=%r", len(queries), streak_info.topic_name)
        return {"candidate_queries": queries}
    except Exception:
        log.exception("QueryPlanner failed — falling back to empty candidate list")
        return {"candidate_queries": []}


def retrieve(state: RAGGraphState) -> dict:
    """Retrieve relevant chunks from the vector store.

    Uses planned candidate queries when available (multi-query retrieval);
    falls back to the legacy single-query path otherwise.
    """
    db = state["db"]
    streak_info = state["streak_info"]
    candidate_queries = state.get("candidate_queries") or []

    retrieval = RAGRetrievalService(db)
    try:
        if len(candidate_queries) > 1:
            chunks = retrieval.retrieve_multi(
                state["topic_id"],
                queries=candidate_queries,
                top_k=5,
            )
        else:
            query = candidate_queries[0] if candidate_queries else streak_info.topic_name
            chunks = retrieval.retrieve(
                state["topic_id"],
                query=query,
                top_k=5,
            )
    finally:
        retrieval.close()

    if not chunks:
        return {
            "failure_reason": "No retrieval chunks found",
            "failure_code": "no_retrieval_chunks",
        }

    return {"retrieved_chunks": chunks}


def reranker(state: RAGGraphState) -> dict:
    """Rerank retrieved chunks by lexical overlap + similarity fusion.

    Best-effort: if the reranker fails the original chunk order is
    preserved so downstream nodes are unaffected.
    """
    chunks = state.get("retrieved_chunks") or []
    if not chunks:
        return {}

    try:
        queries = state.get("candidate_queries") or []
        streak_info = state.get("streak_info")
        if streak_info and streak_info.topic_name:
            queries = [streak_info.topic_name] + queries

        service = RerankerService()
        reranked = service.rerank(chunks, queries)
        log.info("Reranker: %d chunks -> %d reranked", len(chunks), len(reranked))
        return {"retrieved_chunks": reranked}
    except Exception:
        log.exception("Reranker node failed — returning original chunk order")
        return {"retrieved_chunks": list(chunks)}


def context_compressor(state: RAGGraphState) -> dict:
    """Deduplicate and prune low-value chunks after reranking.

    Best-effort: if compression fails the original chunk list is
    preserved so downstream nodes are unaffected.
    """
    chunks = state.get("retrieved_chunks") or []
    if not chunks:
        return {}

    try:
        service = ContextCompressionService()
        compressed = service.compress(chunks)
        log.info("ContextCompressor: %d chunks -> %d compressed", len(chunks), len(compressed))
        return {"retrieved_chunks": compressed}
    except Exception:
        log.exception("ContextCompressor node failed — returning original chunks")
        return {"retrieved_chunks": list(chunks)}


def evidence_gate(state: RAGGraphState) -> dict:
    """Check that compressed chunks provide sufficient evidence for generation.

    If insufficient, sets ``failure_reason`` and ``evidence_sufficient=False``
    so the graph aborts cleanly without persisting.
    """
    chunks = state.get("retrieved_chunks") or []

    try:
        service = EvidenceValidationService()
        report = service.validate(chunks)

        if not report.sufficient:
            log.info("EvidenceGate BLOCKED: %s", report.reason)
            return {
                "evidence_sufficient": False,
                "failure_reason": f"Evidence insufficient: {report.reason}",
                "failure_code": "evidence_insufficient",
            }

        log.info(
            "EvidenceGate PASSED: %d chunks, avg_sim=%.3f, high_quality=%d",
            report.chunk_count,
            report.avg_similarity,
            report.high_quality_count,
        )
        return {"evidence_sufficient": True}
    except Exception:
        log.exception("EvidenceGate node failed — blocking generation as safety fallback")
        return {
            "evidence_sufficient": False,
            "failure_reason": "Evidence gate raised an exception",
            "failure_code": "evidence_gate_exception",
        }


def retrieval_repair(state: RAGGraphState) -> dict:
    """Generate broader, simpler queries and re-enter the retrieval pipeline.

    Bounded: the graph only routes here when ``retry_count < MAX_RETRIES``.
    Clears ``failure_reason`` so downstream nodes don't see stale failure.

    Best-effort: if the repair service raises, the graph ends safely
    with a clear failure reason rather than propagating the exception.
    """
    try:
        streak = state["streak_info"]
        repair = RetrievalRepairService()
        new_queries = repair.repair(
            topic_name=streak.topic_name,
            original_queries=state.get("candidate_queries") or [],
        )
        log.info(
            "RetrievalRepair: %d queries generated (retry %d -> %d)",
            len(new_queries),
            state["retry_count"],
            state["retry_count"] + 1,
        )
        return {
            "retry_count": state["retry_count"] + 1,
            "candidate_queries": new_queries,
            "failure_reason": None,
            "failure_code": None,
            "evidence_sufficient": False,
        }
    except Exception:
        log.exception("RetrievalRepair failed — aborting graph")
        return {
            "retry_count": state["retry_count"] + 1,
            "failure_reason": "Retrieval repair failed",
            "failure_code": "retrieval_repair_failed",
            "evidence_sufficient": False,
        }


def generate(state: RAGGraphState) -> dict:
    """Generate an MCQ from retrieved context.

    When a ``repair_hint`` is present (set by ``question_repair``), adjusts
    the generation parameters accordingly:
      - ``adjusted_theta`` overrides the student ability for difficulty targeting
      - ``context_addendum`` is appended to the generation prompt

    Exception-safe: on failure, sets failure_code so the graph aborts cleanly.
    """
    try:
        db = state["db"]
        streak_info = state["streak_info"]

        # Apply repair hint adjustments
        repair_hint = state.get("repair_hint")
        theta = state["theta"]
        extra_instructions: str | None = None
        if repair_hint:
            adjusted = repair_hint.get("adjusted_theta")
            if adjusted is not None:
                theta = float(adjusted)
                log.info("Generate: repair hint adjusted theta %.3f -> %.3f", state["theta"], theta)
            ctx = repair_hint.get("context_addendum")
            if ctx:
                extra_instructions = str(ctx)

        gen_service = GeneratedQuestionService(db)
        gen_input = GenerationInput(
            topic_id=state["topic_id"],
            topic_name=streak_info.topic_name,
            theta=theta,
            recent_streak=streak_info.current_streak,
            avg_theta=streak_info.avg_theta,
            retrieved_chunks=state["retrieved_chunks"],
            extra_instructions=extra_instructions,
        )
        gen_output = gen_service.generate(gen_input)

        if not gen_output:
            return {
                "failure_reason": "Generation produced no output",
                "failure_code": "generation_no_output",
            }

        return {"gen_output": gen_output}
    except Exception:
        log.exception("Generate node failed — aborting graph")
        return {
            "failure_reason": "Generation raised an exception",
            "failure_code": "generation_exception",
        }


def grounding_validator(state: RAGGraphState) -> dict:
    """Check that the generated question is grounded in retrieved evidence.

    Validates the correct answer text and explanation have sufficient
    lexical support from the retrieved chunks.  Blocks generation when
    the question appears unsupported — fail-closed.

    Exception-safe: on failure, blocks as a safety fallback.
    """
    gen_output = state["gen_output"]
    chunks = state.get("retrieved_chunks") or []

    if not gen_output:
        return {
            "failure_reason": "Grounding check skipped: no gen_output",
            "failure_code": "grounding_skipped_no_output",
        }

    try:
        service = GroundingValidationService()
        report = service.validate(
            question_text=gen_output.question_text,
            correct_answer_text=_extract_correct_answer_text(gen_output),
            explanation=gen_output.explanation,
            retrieved_chunks=chunks,
        )

        if not report.grounded:
            log.info(
                "GroundingValidator BLOCKED: question_ok=%s, answer_ok=%s, "
                "explanation_ok=%s, score=%.3f, issues=%s",
                report.question_supported,
                report.answer_supported,
                report.explanation_supported,
                report.support_score,
                report.issues,
            )
            return {
                "grounding_report": report,
                "failure_reason": (
                    f"Grounding failed: score={report.support_score:.3f}, "
                    f"issues={report.issues}"
                ),
                "failure_code": "grounding_failed",
            }

        log.info(
            "GroundingValidator PASSED: question_ok=%s, answer_ok=%s, "
            "explanation_ok=%s, score=%.3f",
            report.question_supported,
            report.answer_supported,
            report.explanation_supported,
            report.support_score,
        )
        return {"grounding_report": report}
    except Exception:
        log.exception(
            "GroundingValidator node failed — blocking as safety fallback"
        )
        return {
            "failure_reason": "Grounding validator raised an exception",
            "failure_code": "grounding_exception",
        }


def _extract_correct_answer_text(gen_output) -> str:  # type: ignore[no-untyped-def]
    """Extract the correct answer text from GenerationOutput options."""
    for opt in gen_output.options:
        if opt.get("is_correct"):
            return opt.get("text", "")
    return ""


def validate(state: RAGGraphState) -> dict:
    """Run deterministic + LLM-judge validation.

    Exception-safe: on failure, sets failure_code so the graph aborts cleanly.
    """
    try:
        db = state["db"]
        gen_output = state["gen_output"]

        validation = GeneratedQuestionValidationService(db)
        v_report = validation.validate(
            gen_output.question_text,
            gen_output.options,
            gen_output.explanation,
        )

        if not v_report.valid:
            return {
                "validation_report": v_report,
                "failure_reason": f"Validation failed: {v_report.issues}",
                "failure_code": "validation_failed",
            }

        return {"validation_report": v_report}
    except Exception:
        log.exception("Validate node failed — aborting graph")
        return {
            "failure_reason": "Validation raised an exception",
            "failure_code": "validation_exception",
        }


def distractor_validator(state: RAGGraphState) -> dict:
    """Check that generated distractors are distinct, meaningful, and separated from the correct answer.

    Blocks the graph when distractors are weak, duplicated, or too
    similar to the correct answer.

    Exception-safe: on failure, blocks as a safety fallback.
    """
    gen_output = state["gen_output"]

    if not gen_output:
        return {
            "failure_reason": "Distractor check skipped: no gen_output",
            "failure_code": "distractor_skipped_no_output",
        }

    try:
        service = DistractorValidationService()
        report = service.validate(options=gen_output.options)

        if not report.valid:
            log.info(
                "DistractorValidator BLOCKED: distinct=%s, separated=%s, "
                "meaningful=%s, issues=%s",
                report.distinct_distractors,
                report.separated_from_correct,
                report.meaningful_distractors,
                report.issues,
            )
            return {
                "distractor_report": report,
                "failure_reason": (
                    f"Distractor validation failed: issues={report.issues}"
                ),
                "failure_code": "distractor_failed",
            }

        log.info(
            "DistractorValidator PASSED: distinct=%s, separated=%s, "
            "meaningful=%s",
            report.distinct_distractors,
            report.separated_from_correct,
            report.meaningful_distractors,
        )
        return {"distractor_report": report}
    except Exception:
        log.exception(
            "DistractorValidator node failed — blocking as safety fallback"
        )
        return {
            "failure_reason": "Distractor validator raised an exception",
            "failure_code": "distractor_exception",
        }


def difficulty_calibrator(state: RAGGraphState) -> dict:
    """Check that generated question difficulty aligns with target theta.

    Compares the student's current ability (theta) against the generated
    question's difficulty estimate.  Blocks the graph when the question
    is misaligned — too easy or too hard for the student.

    Exception-safe: on failure, blocks as a safety fallback.
    """
    gen_output = state["gen_output"]

    if not gen_output:
        return {
            "failure_reason": "Difficulty calibration skipped: no gen_output",
            "failure_code": "difficulty_skipped_no_output",
        }

    try:
        service = DifficultyCalibrationService()
        report = service.calibrate(
            target_theta=state["theta"],
            difficulty_estimate=gen_output.difficulty_estimate,
        )

        if not report.aligned:
            log.info(
                "DifficultyCalibrator BLOCKED: target=%.3f (%s), "
                "predicted=%s (%s), delta=%s, issues=%s",
                report.target_theta, report.target_band,
                f"{report.predicted_difficulty:.3f}" if report.predicted_difficulty is not None else "None",
                report.predicted_band,
                f"{report.delta:.3f}" if report.delta is not None else "None",
                report.issues,
            )
            return {
                "difficulty_report": report,
                "failure_reason": (
                    f"Difficulty misaligned: delta={report.delta}, "
                    f"issues={report.issues}"
                ),
                "failure_code": "difficulty_misaligned",
            }

        log.info(
            "DifficultyCalibrator PASSED: target=%.3f (%s), "
            "predicted=%.3f (%s), delta=%.3f",
            report.target_theta, report.target_band,
            report.predicted_difficulty, report.predicted_band,
            report.delta,
        )
        return {"difficulty_report": report}
    except Exception:
        log.exception(
            "DifficultyCalibrator node failed — blocking as safety fallback"
        )
        return {
            "failure_reason": "Difficulty calibrator raised an exception",
            "failure_code": "difficulty_exception",
        }


def repair_decision(state: RAGGraphState) -> dict:
    """Decide whether the current failure is repairable and within budget.

    Examines the ``failure_code`` to classify the failure type and
    checks the repair budget.  When repairable, sets a repair hint that
    the ``question_repair`` node forwards to ``generate``.

    Exception-safe: on failure, sets ``repairable=False`` so the graph
    aborts safely (fail-closed).
    """
    failure_reason = state.get("failure_reason")
    failure_code = state.get("failure_code")
    repair_count = state.get("repair_attempt_count", 0)

    # Collect context for difficulty-aware hints
    difficulty_report = state.get("difficulty_report")
    distractor_report = state.get("distractor_report")

    # Compute signed delta (predicted - target) for repair direction.
    # The calibrator's report.delta is always abs(), so we must derive
    # the sign from the raw values stored in the report.
    difficulty_signed_delta: float | None = None
    if (
        difficulty_report is not None
        and difficulty_report.predicted_difficulty is not None
    ):
        difficulty_signed_delta = (
            difficulty_report.predicted_difficulty - difficulty_report.target_theta
        )

    try:
        service = QuestionRepairService()
        report = service.decide(
            failure_code=failure_code,
            failure_reason=failure_reason,
            repair_attempt_count=repair_count,
            target_theta=state.get("theta"),
            difficulty_signed_delta=difficulty_signed_delta,
            distractor_issues=distractor_report.issues if distractor_report else None,
        )

        if report.repairable:
            log.info(
                "RepairDecision ALLOW: type=%s, remaining=%d, hint=%s",
                report.failure_type, report.attempts_remaining, report.hint,
            )
        else:
            log.info(
                "RepairDecision BLOCKED: type=%s, remaining=%d, issues=%s",
                report.failure_type, report.attempts_remaining, report.issues,
            )

        return {"repair_report": report}
    except Exception:
        log.exception("RepairDecision node failed — blocking as safety fallback")
        from app.services.question_repair_service import RepairReport
        return {
            "repair_report": RepairReport(
                repairable=False,
                attempts_remaining=0,
                failure_type="unknown",
                issues=["Repair decision raised an exception"],
            ),
        }


def question_repair(state: RAGGraphState) -> dict:
    """Prepare repair hint and reset state for the next generation pass.

    Increments the repair counter, clears the failure reason, and stores
    the repair hint in state so ``generate`` can read it on the next pass.

    Exception-safe: on failure, sets failure_reason so the graph aborts.
    """
    repair_report = state.get("repair_report")

    if not repair_report or not repair_report.repairable:
        return {
            "failure_reason": "Question repair called but failure is not repairable",
        }

    try:
        new_count = state.get("repair_attempt_count", 0) + 1
        log.info(
            "QuestionRepair: attempt %d, hint_target=%s",
            new_count, repair_report.hint.get("target"),
        )
        return {
            "repair_attempt_count": new_count,
            "repair_hint": repair_report.hint,
            "failure_reason": None,
            "failure_code": None,
            # Clear validator reports so the new generation pass
            # starts fresh — validators will re-evaluate.
            "validation_report": None,
            "grounding_report": None,
            "distractor_report": None,
            "difficulty_report": None,
            "artifact_report": None,
            "gen_output": None,
        }
    except Exception:
        log.exception("QuestionRepair node failed — aborting graph")
        return {
            "failure_reason": "Question repair raised an exception",
        }


def duplicate_gate(state: RAGGraphState) -> dict:
    """Check generated question text against existing questions for duplicates.

    Compares against GeneratedQuestion rows (same topic) and the fixed
    question bank (same topic scope).  Blocks persistence when a
    duplicate is detected.

    Exception-safe: on failure, blocks persistence as a safety fallback.
    """
    gen_output = state["gen_output"]
    db = state["db"]

    try:
        service = QuestionDedupService()
        report = service.check(
            db,
            topic_id=state["topic_id"],
            question_text=gen_output.question_text,
        )

        if report.is_duplicate:
            log.info(
                "DuplicateGate BLOCKED: max_sim=%.3f, source=%s, compared=%d",
                report.max_similarity,
                report.source,
                report.compared_count,
            )
            return {
                "failure_reason": (
                    f"Duplicate detected: similarity={report.max_similarity:.3f}, "
                    f"source={report.source}"
                ),
                "failure_code": "duplicate_detected",
            }

        log.info(
            "DuplicateGate PASSED: max_sim=%.3f, compared=%d",
            report.max_similarity,
            report.compared_count,
        )
        return {}
    except Exception:
        log.exception("DuplicateGate node failed — blocking persistence as safety fallback")
        return {
            "failure_reason": "Duplicate gate raised an exception",
            "failure_code": "duplicate_exception",
        }


def confidence_gate(state: RAGGraphState) -> dict:
    """Route generated question based on deterministic confidence scoring.

    Evaluates evidence quality, validation signals, retrieval context,
    and question completeness to decide:
      - ``auto_approve``: persist as-is
      - ``human_review``: persist as draft / review_required
      - ``reject``: do not persist

    If ``RAG_REVIEW_REQUIRED`` is globally enabled, ``auto_approve`` is
    overridden to ``human_review`` at persist time (not here).

    Exception-safe: on failure, routes to human review as a conservative
    safety fallback so no work is silently lost.
    """
    gen_output = state["gen_output"]
    chunks = state.get("retrieved_chunks") or []
    v_report = state.get("validation_report")

    try:
        service = QuestionConfidenceService()
        report = service.evaluate(
            retrieved_chunks=chunks,
            validation_report=v_report,
            retry_count=state.get("retry_count", 0),
            gen_output=gen_output,
        )

        if report.route == "reject":
            log.info(
                "ConfidenceGate REJECT: score=%.1f, reasons=%s",
                report.score,
                report.reasons,
            )
            return {
                "confidence_report": report,
                "failure_reason": (
                    f"Confidence rejected: score={report.score:.1f}"
                ),
                "failure_code": "confidence_rejected",
            }

        log.info(
            "ConfidenceGate PASS: route=%s, score=%.1f, reasons=%s",
            report.route,
            report.score,
            report.reasons,
        )
        return {"confidence_report": report}
    except Exception:
        log.exception(
            "ConfidenceGate node failed — routing to human review as safety fallback"
        )
        fallback = ConfidenceReport(
            route="human_review",
            score=0.0,
            reasons=["Confidence gate exception — safety fallback to human review"],
        )
        return {"confidence_report": fallback}


def artifact_validator(state: RAGGraphState) -> dict:
    """Validate evidence citations and distractor rationale as structured artifacts.

    Deterministic, no LLM judge.  When the config flags
    ``GENERATED_MCQ_REQUIRE_CITATIONS`` / ``GENERATED_MCQ_REQUIRE_DISTRACTOR_RATIONALE``
    are ``False``, the respective checks pass silently.

    Exception-safe: on failure, blocks persistence as a safety fallback.
    """
    gen_output = state.get("gen_output")
    chunks = state.get("retrieved_chunks") or []

    if not gen_output:
        return {
            "failure_reason": "Artifact validation skipped: no gen_output",
            "failure_code": "artifact_skipped_no_output",
        }

    try:
        settings = get_settings()
        service = GeneratedArtifactValidationService(
            require_citations=settings.GENERATED_MCQ_REQUIRE_CITATIONS,
            require_rationale=settings.GENERATED_MCQ_REQUIRE_DISTRACTOR_RATIONALE,
        )
        report = service.validate(
            evidence_citations=gen_output.evidence_citations,
            distractor_rationale=gen_output.distractor_rationale,
            options=gen_output.options,
            retrieved_chunks=chunks,
        )

        if not report.citations_valid or not report.rationale_valid:
            log.info(
                "ArtifactValidator BLOCKED: citations_ok=%s, rationale_ok=%s, issues=%s",
                report.citations_valid,
                report.rationale_valid,
                report.issues,
            )
            return {
                "artifact_report": report,
                "failure_reason": (
                    f"Artifact validation failed: issues={report.issues}"
                ),
                "failure_code": "artifact_failed",
            }

        log.info(
            "ArtifactValidator PASSED: citations_ok=%s, rationale_ok=%s",
            report.citations_valid,
            report.rationale_valid,
        )
        return {"artifact_report": report}
    except Exception:
        log.exception(
            "ArtifactValidator node failed — blocking persistence as safety fallback"
        )
        return {
            "failure_reason": "Artifact validator raised an exception",
            "failure_code": "artifact_exception",
        }


def persist(state: RAGGraphState) -> dict:
    """Persist generated question, evidence rows, and update streak counter.

    Exception-safe: on failure, sets failure_code so the graph aborts cleanly.
    """
    try:
        settings = get_settings()
        db = state["db"]
        gen_output = state["gen_output"]
        v_report = state["validation_report"]
        chunks = state["retrieved_chunks"]

        gq = _create_generated_question(
            db, state, gen_output, v_report, settings,
        )
        _create_evidence_rows(db, gq.id, chunks)
        _increment_streak(db, state)

        log.info(
            "LangGraph: Persisted %d evidence rows for question %d",
            len(chunks),
            gq.id,
        )
        return {"generated_question_id": gq.id}
    except Exception:
        log.exception("Persist node failed — aborting graph")
        return {
            "failure_reason": "Persist raised an exception",
            "failure_code": "persist_exception",
        }


def _create_generated_question(db, state, gen_output, v_report, settings):  # type: ignore[no-untyped-def]
    from app.models.rag import GeneratedQuestion
    from app.services.rag_telemetry_service import build_langgraph_trace

    confidence_report = state.get("confidence_report")
    raw_route = confidence_report.route if confidence_report else "auto_approve"

    if raw_route == "human_review" or settings.RAG_REVIEW_REQUIRED:
        status = GeneratedQuestionStatus.draft
        review_required = True
        effective_route = "human_review"
    else:
        status = GeneratedQuestionStatus.auto_approved
        review_required = False
        effective_route = "auto_approve"

    vr_dict = {
        "schema_ok": v_report.schema_ok,
        "single_correct": v_report.single_correct,
        "non_duplicate": v_report.non_duplicate,
        "max_similarity": v_report.max_similarity,
        "issues": v_report.issues,
        "confidence_route_raw": raw_route,
        "confidence_route_effective": effective_route,
        "confidence_score": confidence_report.score if confidence_report else None,
        "judge_feedback": v_report.judge_feedback,
        "judge_ok": v_report.judge_ok,
        "judge_ambiguity": v_report.judge_ambiguity,
        "judge_factual_error": v_report.judge_factual_error,
    }

    grounding_report = state.get("grounding_report")
    if grounding_report is not None:
        vr_dict["grounding"] = {
            "grounded": grounding_report.grounded,
            "question_supported": grounding_report.question_supported,
            "answer_supported": grounding_report.answer_supported,
            "explanation_supported": grounding_report.explanation_supported,
            "support_score": grounding_report.support_score,
            "issues": grounding_report.issues,
        }

    distractor_report = state.get("distractor_report")
    if distractor_report is not None:
        vr_dict["distractor"] = {
            "valid": distractor_report.valid,
            "distinct_distractors": distractor_report.distinct_distractors,
            "separated_from_correct": distractor_report.separated_from_correct,
            "meaningful_distractors": distractor_report.meaningful_distractors,
            "issues": distractor_report.issues,
        }

    difficulty_report = state.get("difficulty_report")
    if difficulty_report is not None:
        vr_dict["difficulty"] = {
            "aligned": difficulty_report.aligned,
            "target_theta": difficulty_report.target_theta,
            "predicted_difficulty": difficulty_report.predicted_difficulty,
            "delta": difficulty_report.delta,
            "target_band": difficulty_report.target_band,
            "predicted_band": difficulty_report.predicted_band,
            "issues": difficulty_report.issues,
        }

    repair_attempt_count = state.get("repair_attempt_count", 0)
    if repair_attempt_count > 0:
        vr_dict["repair"] = {
            "attempt_count": repair_attempt_count,
            "final_hint_target": state.get("repair_hint", {}).get("target") if state.get("repair_hint") else None,
        }

    artifact_report = state.get("artifact_report")
    if artifact_report is not None:
        vr_dict["artifact"] = {
            "citations_valid": artifact_report.citations_valid,
            "rationale_valid": artifact_report.rationale_valid,
            "issues": artifact_report.issues,
        }

    gen_output = state["gen_output"]
    if gen_output.evidence_citations is not None:
        vr_dict["evidence_citations"] = gen_output.evidence_citations
    if gen_output.distractor_rationale is not None:
        vr_dict["distractor_rationale"] = gen_output.distractor_rationale

    try:
        langgraph_trace = build_langgraph_trace(state, settings)
        vr_dict["langgraph_trace"] = langgraph_trace
    except Exception:
        log.warning("build_langgraph_trace failed — persisting without trace")

    gq = GeneratedQuestion(
        topic_id=state["topic_id"],
        source_exam_id=state["exam_id"],
        text=gen_output.question_text,
        choices=gen_output.options,
        explanation=gen_output.explanation,
        difficulty_estimate=gen_output.difficulty_estimate,
        status=status,
        review_required=review_required,
        validation_report=vr_dict,
    )
    db.add(gq)
    db.flush()
    return gq


def _create_evidence_rows(db, question_id: int, chunks: list) -> None:  # type: ignore[no-untyped-def]
    from app.models.rag import GeneratedQuestionEvidence

    for rc in chunks:
        evidence = GeneratedQuestionEvidence(
            generated_question_id=question_id,
            chunk_id=rc.chunk_id,
            relevance_score=rc.similarity,
        )
        db.add(evidence)
    db.flush()


def _increment_streak(db, state: RAGGraphState) -> None:
    streak_service = TopicStreakService(db, state["student_id"])
    streak_service.increment_generated(state["exam_id"], state["topic_id"])


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

MAX_RETRIES: int = 1


def _after_gate_check(state: RAGGraphState) -> str:
    if state.get("failure_reason"):
        return END
    return "query_planner"


def _after_query_planner(state: RAGGraphState) -> str:
    """Always proceed to retrieve — planner is best-effort."""
    return "retrieve"


def _after_retrieve(state: RAGGraphState) -> str:
    if state.get("failure_reason"):
        return END
    return "reranker"


def _after_reranker(state: RAGGraphState) -> str:
    """Always proceed to context_compressor — reranker is best-effort."""
    return "context_compressor"


def _after_context_compressor(state: RAGGraphState) -> str:
    """Always proceed to evidence_gate — compressor is best-effort."""
    return "evidence_gate"


def _after_evidence_gate(state: RAGGraphState) -> str:
    if not state.get("failure_reason"):
        return "generate"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return END
    return "retrieval_repair"


def _after_generate(state: RAGGraphState) -> str:
    if state.get("failure_reason"):
        return END
    return "grounding_validator"


def _after_grounding_validator(state: RAGGraphState) -> str:
    if state.get("failure_reason"):
        return END
    return "validate"


def _after_validate(state: RAGGraphState) -> str:
    if state.get("failure_reason"):
        return END
    return "distractor_validator"


def _after_distractor_validator(state: RAGGraphState) -> str:
    """Always proceed to difficulty_calibrator.

    The difficulty calibrator checks failure_reason and may route to
    repair_decision for repairable failures.  Terminal failures
    (evidence, grounding, duplicate, generation) are already handled
    before this point.
    """
    return "difficulty_calibrator"


def _after_difficulty_calibrator(state: RAGGraphState) -> str:
    if state.get("failure_reason"):
        return "repair_decision"
    return "duplicate_gate"


def _after_repair_decision(state: RAGGraphState) -> str:
    repair_report = state.get("repair_report")
    if repair_report and repair_report.repairable:
        return "question_repair"
    return END


def _after_question_repair(state: RAGGraphState) -> str:
    """After repair prep, route back to generate for a fresh attempt."""
    if state.get("failure_reason"):
        return END
    return "generate"


def _after_duplicate_gate(state: RAGGraphState) -> str:
    if state.get("failure_reason"):
        return END
    return "confidence_gate"


def _after_confidence_gate(state: RAGGraphState) -> str:
    if state.get("failure_reason"):
        return END
    return "artifact_validator"


def _after_artifact_validator(state: RAGGraphState) -> str:
    if state.get("failure_reason"):
        return END
    return "persist"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

_GRAPH_NODES = [
    ("gate_check", gate_check),
    ("query_planner", query_planner),
    ("retrieve", retrieve),
    ("reranker", reranker),
    ("context_compressor", context_compressor),
    ("evidence_gate", evidence_gate),
    ("retrieval_repair", retrieval_repair),
    ("generate", generate),
    ("grounding_validator", grounding_validator),
    ("validate", validate),
    ("distractor_validator", distractor_validator),
    ("difficulty_calibrator", difficulty_calibrator),
    ("repair_decision", repair_decision),
    ("question_repair", question_repair),
    ("duplicate_gate", duplicate_gate),
    ("confidence_gate", confidence_gate),
    ("artifact_validator", artifact_validator),
    ("persist", persist),
]

_CONDITIONAL_EDGES = [
    ("gate_check", _after_gate_check, {"query_planner": "query_planner", END: END}),
    ("query_planner", _after_query_planner, {"retrieve": "retrieve"}),
    ("retrieve", _after_retrieve, {"reranker": "reranker", END: END}),
    ("reranker", _after_reranker, {"context_compressor": "context_compressor"}),
    ("context_compressor", _after_context_compressor, {"evidence_gate": "evidence_gate"}),
    ("evidence_gate", _after_evidence_gate, {"generate": "generate", "retrieval_repair": "retrieval_repair", END: END}),
    ("generate", _after_generate, {"grounding_validator": "grounding_validator", END: END}),
    ("grounding_validator", _after_grounding_validator, {"validate": "validate", END: END}),
    ("validate", _after_validate, {"distractor_validator": "distractor_validator", END: END}),
    ("distractor_validator", _after_distractor_validator, {"difficulty_calibrator": "difficulty_calibrator", END: END}),
    ("difficulty_calibrator", _after_difficulty_calibrator, {"repair_decision": "repair_decision", "duplicate_gate": "duplicate_gate"}),
    ("repair_decision", _after_repair_decision, {"question_repair": "question_repair", END: END}),
    ("question_repair", _after_question_repair, {"generate": "generate", END: END}),
    ("duplicate_gate", _after_duplicate_gate, {"confidence_gate": "confidence_gate", END: END}),
    ("confidence_gate", _after_confidence_gate, {"artifact_validator": "artifact_validator", END: END}),
    ("artifact_validator", _after_artifact_validator, {"persist": "persist", END: END}),
]


def _build_compiled_graph():  # type: ignore[no-untyped-def]
    graph = StateGraph(RAGGraphState)

    for name, fn in _GRAPH_NODES:
        graph.add_node(name, fn)

    graph.set_entry_point("gate_check")

    for source, condition, path_map in _CONDITIONAL_EDGES:
        graph.add_conditional_edges(source, condition, path_map)

    graph.add_edge("persist", END)
    graph.add_edge("retrieval_repair", "retrieve")

    return graph.compile()


_compiled_graph = None


def _get_compiled_graph():  # type: ignore[no-untyped-def]
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_compiled_graph()
    return _compiled_graph


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_rag_graph(
    db,  # type: ignore[no-untyped-def]
    exam_id: int,
    topic_id: int,
    theta: float,
    student_id: int,  # noqa: ARG001 — kept for future state enrichment
):
    """Execute the LangGraph RAG workflow.

    Returns the persisted :class:`GeneratedQuestion` on success, or ``None``
    when the graph exits early (gate check, retrieval, generation, or
    validation failure).

    The DB session is **not** committed here — the caller
    (``submit_adaptive_answer``) retains commit ownership.
    """
    from app.models.rag import GeneratedQuestion

    graph = _get_compiled_graph()

    trace_id = uuid4().hex

    initial_state: RAGGraphState = {
        "db": db,
        "exam_id": exam_id,
        "student_id": student_id,
        "topic_id": topic_id,
        "theta": theta,
        "streak_info": None,
        "candidate_queries": [],
        "retrieved_chunks": [],
        "gen_output": None,
        "validation_report": None,
        "grounding_report": None,
        "distractor_report": None,
        "difficulty_report": None,
        "artifact_report": None,
        "repair_report": None,
        "repair_attempt_count": 0,
        "repair_hint": None,
        "confidence_report": None,
        "generated_question_id": None,
        "evidence_sufficient": False,
        "failure_reason": None,
        "failure_code": None,
        "retry_count": 0,
        "trace_id": trace_id,
    }

    result = graph.invoke(initial_state)

    qid = result.get("generated_question_id")
    if qid is None:
        reason = result.get("failure_reason", "unknown")
        log.info("LangGraph RAG graph ended without generation: %s", reason)
        return None

    return db.query(GeneratedQuestion).filter(GeneratedQuestion.id == qid).first()

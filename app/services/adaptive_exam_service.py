from __future__ import annotations

import math
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models import AdaptiveExam, AdaptiveExamResponse, Choice, ExamStatus, GeneratedQuestion, Phase, Question, StudentTopicProgress, SubTopic, Topic
from app.models.enums import CognitiveLevel, DifficultyLevel, GeneratedQuestionStatus
from app.services.generated_question_service import GeneratedQuestionService, GenerationInput
from app.services.generated_question_validation_service import GeneratedQuestionValidationService
from app.services.question_dedup_service import QuestionDedupService
from app.services.rag_retrieval_service import RAGRetrievalService
from app.services.topic_streak_service import TopicStreakService

log = get_logger(__name__)

try:
    from catsim.selection import MaxInfoSelector  # type: ignore
    _HAS_CATSIM = True
except Exception:
    _HAS_CATSIM = False


def _seed_irt_values(question: Question) -> None:
    if question.irt_a is None:
        question.irt_a = 1.0
    if question.irt_c is None:
        question.irt_c = 0.2
    if question.irt_b is None:
        diff = str(question.difficulty.value if hasattr(question.difficulty, "value") else question.difficulty)
        question.irt_b = -1.0 if diff == "Easy" else (0.0 if diff == "Medium" else 1.0)


def _get_phase_question_pool(db: Session, phase_id: int, seed_missing_irt: bool = False) -> list[Question]:
    cc = (
        select(Choice.question_id.label("qid"), func.count(Choice.id).label("cnt"))
        .where(Choice.is_correct == True)  # noqa: E712
        .group_by(Choice.question_id)
        .subquery()
    )
    rows = (
        db.query(Question)
        .join(SubTopic, Question.subtopic_id == SubTopic.id)
        .join(Topic, SubTopic.topic_id == Topic.id)
        .join(Phase, Topic.phase_id == Phase.id)
        .join(cc, cc.c.qid == Question.id)
        .filter(
            Phase.id == phase_id,
            Question.is_active == True,  # noqa: E712
            cc.c.cnt == 1,
        )
        .all()
    )
    if seed_missing_irt:
        dirty = False
        for q in rows:
            if q.irt_a is None or q.irt_b is None or q.irt_c is None:
                _seed_irt_values(q)
                db.add(q)
                dirty = True
        if dirty:
            db.flush()
    return rows


def _item_information(theta: float, a: float, b: float, c: float) -> float:
    c = max(0.0, min(0.35, c))
    a = max(0.2, a)
    p = c + (1.0 - c) / (1.0 + math.exp(-a * (theta - b)))
    if p <= c or p >= 1.0:
        return 0.0
    num = (a ** 2) * ((p - c) ** 2) * (1.0 - p)
    den = ((1.0 - c) ** 2) * p
    return num / den if den > 0 else 0.0


def _fallback_select_next(questions: list[Question], asked_ids: set[int], theta: float) -> Question | None:
    best: Question | None = None
    best_info = -1.0
    for q in questions:
        if q.id in asked_ids:
            continue
        a = float(q.irt_a or 1.0)
        b = float(q.irt_b or 0.0)
        c = float(q.irt_c or 0.2)
        info = _item_information(theta, a, b, c)
        if info > best_info:
            best_info = info
            best = q
    return best


def _select_next_question(questions: list[Question], asked_ids: set[int], theta: float) -> Question | None:
    remaining = [q for q in questions if q.id not in asked_ids]
    if not remaining:
        return None

    if _HAS_CATSIM:
        try:
            matrix = [[float(q.irt_a or 1.0), float(q.irt_b or 0.0), float(q.irt_c or 0.2), 1.0] for q in questions]
            id_to_index = {q.id: i for i, q in enumerate(questions)}
            administered = [id_to_index[qid] for qid in asked_ids if qid in id_to_index]
            idx = MaxInfoSelector().select(items=matrix, administered_items=administered, est_theta=theta)
            if isinstance(idx, int) and 0 <= idx < len(questions):
                pick = questions[idx]
                if pick.id not in asked_ids:
                    return pick
        except Exception:
            pass

    return _fallback_select_next(questions, asked_ids, theta)


def _next_theta(theta_before: float, is_correct: bool) -> float:
    step = 0.35 if is_correct else -0.35
    return max(-4.0, min(4.0, theta_before + step))


def _select_next_from_topic(
    pool: list[Question],
    asked_ids: set[int],
    topic_id: int,
    theta: float,
) -> Question | None:
    """Select the best unasked question from a specific topic.

    Used when the exam is locked to a topic during the streak-building
    phase.  Filters the pool to the given topic, then applies the
    standard IRT selection.
    """
    topic_qs = [q for q in pool if q.subtopic.topic_id == topic_id and q.id not in asked_ids]
    if not topic_qs:
        return None
    return _select_next_question(topic_qs, set(), theta)


def _question_payload(db: Session, q: Question) -> dict:
    choices = db.query(Choice).filter(Choice.question_id == q.id).order_by(Choice.id.asc()).all()
    topic = db.query(Topic).filter(Topic.id == q.subtopic.topic_id).first()
    return {
        "id": q.id,
        "text": q.text,
        "difficulty": q.difficulty,
        "cognitive_level": q.cognitive_level,
        "question_type": q.question_type,
        "is_generated": False,
        "topic_id": topic.id if topic else 0,
        "topic_name": topic.name if topic else "Unknown",
        "choices": [{"id": c.id, "text": c.text} for c in choices],
    }


def _map_difficulty_estimate(estimate: float | None) -> DifficultyLevel:
    if estimate is None:
        return DifficultyLevel.Medium
    if estimate < -0.5:
        return DifficultyLevel.Easy
    if estimate > 0.5:
        return DifficultyLevel.Hard
    return DifficultyLevel.Medium


def _generated_question_payload(db: Session, gq: GeneratedQuestion) -> dict:
    topic = db.query(Topic).filter(Topic.id == gq.topic_id).first()
    return {
        "id": gq.id,
        "text": gq.text,
        "difficulty": _map_difficulty_estimate(gq.difficulty_estimate),
        "cognitive_level": CognitiveLevel.Application,
        "question_type": "SingleChoice",
        "is_generated": True,
        "topic_id": topic.id if topic else 0,
        "topic_name": topic.name if topic else "Unknown",
        "choices": [
            {"id": i, "text": opt["text"]}
            for i, opt in enumerate(gq.choices, start=1)
        ],
    }


def _build_result(exam: AdaptiveExam, answered: list[AdaptiveExamResponse]) -> dict:
    settings = get_settings()
    total = len(answered)
    correct = sum(1 for x in answered if x.is_correct)
    score = round((correct / max(total, 1)) * 100, 2)
    exam.score = score
    exam.status = ExamStatus.Completed
    exam.submitted_at = datetime.now(timezone.utc)
    passed = score >= float(settings.PHASE2_PASSING_SCORE)
    return {
        "score_percent": float(score),
        "correct_count": correct,
        "total_questions": total,
        "passed": passed,
        "final_theta": float(exam.current_theta),
        "message": "Phase II adaptive exam passed" if passed else "Phase II adaptive exam not passed",
    }


def _progress_payload(
    db: Session,
    exam: AdaptiveExam,
    next_question: Question | None = None,
    result: dict | None = None,
    next_generated: GeneratedQuestion | None = None,
) -> dict:
    if next_generated:
        question_payload = _generated_question_payload(db, next_generated)
    elif next_question:
        question_payload = _question_payload(db, next_question)
    else:
        question_payload = None
    return {
        "exam_id": exam.id,
        "status": exam.status,
        "phase_id": exam.phase_id,
        "answered_count": exam.answered_count,
        "max_questions": exam.max_questions,
        "remaining_questions": max(exam.max_questions - exam.answered_count, 0),
        "current_theta": float(exam.current_theta),
        "current_question": question_payload,
        "result": result,
        "started_at": exam.started_at,
        "submitted_at": exam.submitted_at,
        "current_question_started_at": exam.current_question_started_at,
    }


def _mark_question_served(
    exam: AdaptiveExam,
    *,
    question_id: int | None = None,
    generated_question_id: int | None = None,
) -> None:
    """Record which question the current serve timer belongs to.

    Must be called before commit whenever a question is served so the
    backend can later attribute measured elapsed time to the exact
    question being answered.  The timestamp is always the serve moment
    (UTC) — it is never invented or backfilled elsewhere.
    """
    exam.current_question_id = question_id
    exam.current_generated_question_id = generated_question_id
    exam.current_question_started_at = datetime.now(timezone.utc)


def _clear_current_question(exam: AdaptiveExam) -> None:
    """Clear the tracked served-question state (identity + timestamp)."""
    exam.current_question_id = None
    exam.current_generated_question_id = None
    exam.current_question_started_at = None


def start_adaptive_exam(db: Session, student_id: int, phase_id: int | None) -> dict:
    settings = get_settings()
    if not settings.PHASE2_ENABLED:
        raise ValueError("Phase II adaptive exam is disabled")

    target_phase_id = phase_id or settings.PHASE2_PHASE_ID
    if target_phase_id != settings.PHASE2_PHASE_ID:
        raise ValueError("Only Phase II adaptive exam is supported in this flow")

    phase = db.query(Phase).filter(Phase.id == target_phase_id).first()
    if not phase:
        raise ValueError("Phase not found")

    active = (
        db.query(AdaptiveExam)
        .filter(
            AdaptiveExam.student_id == student_id,
            AdaptiveExam.phase_id == target_phase_id,
            AdaptiveExam.status.in_([ExamStatus.Pending, ExamStatus.InProgress]),
        )
        .first()
    )
    if active:
        raise ValueError("You already have an active Phase II adaptive exam")

    pool = _get_phase_question_pool(db, target_phase_id, seed_missing_irt=True)
    if not pool:
        raise ValueError("No eligible questions found for Phase II adaptive exam")

    max_q = min(int(settings.PHASE2_MAX_QUESTIONS), len(pool))
    exam = AdaptiveExam(
        student_id=student_id,
        phase_id=target_phase_id,
        status=ExamStatus.InProgress,
        max_questions=max_q,
        answered_count=0,
        current_theta=float(settings.PHASE2_INITIAL_THETA),
    )
    db.add(exam)
    db.flush()

    first_question = _select_next_question(pool, set(), exam.current_theta)
    if not first_question:
        raise ValueError("Could not initialize adaptive exam")

    exam.current_question_id = first_question.id
    exam.current_generated_question_id = None
    exam.current_question_started_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(exam)
    return _progress_payload(db, exam, first_question, result=None)


def _is_generated_question(db: Session, question_id: int, exam_id: int) -> GeneratedQuestion | None:
    return (
        db.query(GeneratedQuestion)
        .filter(GeneratedQuestion.id == question_id, GeneratedQuestion.source_exam_id == exam_id)
        .first()
    )


def _try_generate_next(db: Session, exam_id: int, topic_id: int, theta: float, student_id: int) -> GeneratedQuestion | None:
    settings = get_settings()
    if not settings.RAG_ENABLED:
        return None

    # --- LangGraph path (feature-flagged) ---
    if settings.RAG_LANGGRAPH_ENABLED:
        from app.services.langgraph_rag_workflow import run_rag_graph

        return run_rag_graph(db, exam_id, topic_id, theta, student_id)

    # --- Legacy inline path (default) ---
    streak_service = TopicStreakService(db, student_id)
    streak_info = streak_service.get_streak(exam_id, topic_id)
    if not streak_info.can_generate:
        return None

    retrieval = RAGRetrievalService(db)
    chunks = retrieval.retrieve(topic_id, query=streak_info.topic_name, top_k=5)
    retrieval.close()
    if not chunks:
        log.info("No retrieval chunks for topic %d, skipping generation", topic_id)
        return None

    gen_service = GeneratedQuestionService(db)
    gen_input = GenerationInput(
        topic_id=topic_id,
        topic_name=streak_info.topic_name,
        theta=theta,
        recent_streak=streak_info.current_streak,
        avg_theta=streak_info.avg_theta,
        retrieved_chunks=chunks,
    )
    gen_output = gen_service.generate(gen_input)
    if not gen_output:
        log.info("Generation produced no output for topic %d", topic_id)
        return None

    validation = GeneratedQuestionValidationService(db)
    v_report = validation.validate(gen_output.question_text, gen_output.options, gen_output.explanation)
    if not v_report.valid:
        log.warning("Generated question failed validation: %s", v_report.issues)
        return None

    dedup = QuestionDedupService()
    dedup_report = dedup.check(db, topic_id=topic_id, question_text=gen_output.question_text)
    if dedup_report.is_duplicate:
        log.warning(
            "Generated question is a duplicate: max_sim=%.3f, source=%s",
            dedup_report.max_similarity, dedup_report.source,
        )
        return None

    gq = GeneratedQuestion(
        topic_id=topic_id,
        source_exam_id=exam_id,
        text=gen_output.question_text,
        choices=gen_output.options,
        explanation=gen_output.explanation,
        difficulty_estimate=gen_output.difficulty_estimate,
        status=GeneratedQuestionStatus.auto_approved if not settings.RAG_REVIEW_REQUIRED else GeneratedQuestionStatus.draft,
        review_required=bool(settings.RAG_REVIEW_REQUIRED),
        validation_report={
            "schema_ok": v_report.schema_ok,
            "single_correct": v_report.single_correct,
            "non_duplicate": v_report.non_duplicate,
            "max_similarity": v_report.max_similarity,
            "issues": v_report.issues,
            "judge_feedback": v_report.judge_feedback,
            "judge_ok": v_report.judge_ok,
            "judge_ambiguity": v_report.judge_ambiguity,
            "judge_factual_error": v_report.judge_factual_error,
        },
    )
    db.add(gq)
    db.flush()

    # Persist source evidence for every retrieved chunk used in generation
    from app.models.rag import GeneratedQuestionEvidence
    for rc in chunks:
        evidence = GeneratedQuestionEvidence(
            generated_question_id=gq.id,
            chunk_id=rc.chunk_id,
            relevance_score=rc.similarity,
        )
        db.add(evidence)
    db.flush()
    log.info("Persisted %d evidence rows for question %d", len(chunks), gq.id)

    streak_service.increment_generated(exam_id, topic_id)
    log.info("Generated question %d for exam %d topic %d", gq.id, exam_id, topic_id)
    return gq


def submit_adaptive_answer(
    db: Session,
    student_id: int,
    exam_id: int,
    question_id: int,
    choice_id: int,
) -> dict:
    exam = db.query(AdaptiveExam).filter(AdaptiveExam.id == exam_id, AdaptiveExam.student_id == student_id).first()
    if not exam:
        raise ValueError("Adaptive exam not found")
    if exam.status == ExamStatus.Completed:
        raise ValueError("Adaptive exam already completed")

    pool = _get_phase_question_pool(db, exam.phase_id, seed_missing_irt=False)
    by_id = {q.id: q for q in pool}
    q = by_id.get(question_id)
    generated_q = None

    if not q:
        generated_q = _is_generated_question(db, question_id, exam_id)
        if not generated_q:
            raise ValueError("Question does not belong to this adaptive exam phase")

    if q:
        exists = (
            db.query(AdaptiveExamResponse)
            .filter(AdaptiveExamResponse.adaptive_exam_id == exam.id, AdaptiveExamResponse.question_id == question_id)
            .first()
        )
        if exists:
            raise ValueError("Question already answered")

        choice = db.query(Choice).filter(Choice.id == choice_id, Choice.question_id == question_id).first()
        if not choice:
            raise ValueError("Invalid choice for this question")

        correct_choice = db.query(Choice.id).filter(Choice.question_id == question_id, Choice.is_correct == True).first()  # noqa: E712
        is_correct = bool(correct_choice and int(correct_choice[0]) == choice_id)
        topic_id = q.subtopic.topic_id
    else:
        exists = (
            db.query(AdaptiveExamResponse)
            .filter(
                AdaptiveExamResponse.adaptive_exam_id == exam.id,
                AdaptiveExamResponse.generated_question_id == question_id,
            )
            .first()
        )
        if exists:
            raise ValueError("Question already answered")

        selected_idx = choice_id - 1
        options = generated_q.choices
        if selected_idx < 0 or selected_idx >= len(options):
            raise ValueError("Invalid choice for this question")

        is_correct = bool(options[selected_idx].get("is_correct"))
        topic_id = generated_q.topic_id

    theta_before = float(exam.current_theta)
    theta_after = _next_theta(theta_before, is_correct)

    # ── Server-side elapsed time (authoritative, never trusts client) ──
    # Trusted only when the exact served question identity matches the
    # submitted question AND the server-owned serve timestamp exists.
    # Otherwise timing is marked untrusted and anomaly scoring is skipped
    # — never guess or backfill a serve time.
    now = datetime.now(timezone.utc)
    tracked_id = exam.current_question_id if q is not None else exam.current_generated_question_id
    if tracked_id is not None and tracked_id != question_id:
        timing_trusted = False
        timing_issue = "question_mismatch"
        server_elapsed = None
    elif tracked_id is None:
        other_tracked = (
            exam.current_generated_question_id if q is not None else exam.current_question_id
        )
        timing_trusted = False
        timing_issue = "question_mismatch" if other_tracked is not None else "no_tracked_question"
        server_elapsed = None
    elif exam.current_question_started_at is None:
        timing_trusted = False
        timing_issue = "missing_serve_timestamp"
        server_elapsed = None
    else:
        timing_trusted = True
        timing_issue = None
        server_elapsed = max((now - exam.current_question_started_at).total_seconds(), 0.0)

    ans = AdaptiveExamResponse(
        adaptive_exam_id=exam.id,
        question_id=question_id if q else None,
        generated_question_id=generated_q.id if generated_q else None,
        choice_id=choice_id if q else None,
        selected_option_index=None if q else choice_id,
        order_index=exam.answered_count + 1,
        is_correct=is_correct,
        theta_before=theta_before,
        theta_after=theta_after,
        elapsed_seconds=server_elapsed,
        timing_trusted=timing_trusted,
        timing_issue=timing_issue,
    )

    # ── Anomaly detection (Phase II correct answers only, regular Qs,
    #    and only when timing is trusted) ──
    if is_correct and q is not None and timing_trusted:
        try:
            from app.services.anomaly_detection_service import score_response

            irt_b = float(q.irt_b) if q.irt_b is not None else 0.0
            result = score_response(
                student_ability=theta_before,
                question_difficulty=irt_b,
                elapsed_seconds=server_elapsed,
            )
            if result is not None:
                ans.anomaly_flag = bool(result["anomaly_flag"])
                ans.anomaly_score = result["anomaly_score"]
                ans.predicted_class = result["predicted_class"]
                ans.response_interpretation = result["response_interpretation"]
            else:
                log.warning(
                    "Anomaly scoring returned None for exam=%d question=%d, skipping persistence",
                    exam.id, question_id,
                )
        except Exception:
            log.exception(
                "Anomaly detection hook failed for exam=%d question=%d — exam flow continues",
                exam.id, question_id,
            )
    elif is_correct and q is not None:
        log.warning(
            "Anomaly scoring skipped for exam=%d question=%d — untrusted timing (%s)",
            exam.id, question_id, timing_issue,
        )

    db.add(ans)

    exam.answered_count += 1
    exam.current_theta = theta_after
    db.add(exam)
    db.flush()

    ts_service = TopicStreakService(db, student_id)
    ts_service.update_topic_theta(exam_id, topic_id, theta_after)

    answered = (
        db.query(AdaptiveExamResponse)
        .filter(AdaptiveExamResponse.adaptive_exam_id == exam.id)
        .order_by(AdaptiveExamResponse.order_index.asc())
        .all()
    )
    asked_ids = {r.question_id for r in answered if r.question_id is not None}

    # ── Branch: answered a GENERATED question ──────────────────────
    if generated_q:
        ts_service.mark_topic_consumed(exam_id, topic_id)
        exam.locked_topic_id = None
        exam.pending_generated_question_id = None
        db.add(exam)
        db.flush()

        if exam.answered_count >= exam.max_questions:
            _clear_current_question(exam)
            result = _build_result(exam, answered)
            db.add(exam)
            db.commit()
            db.refresh(exam)
            return _progress_payload(db, exam, result=result)

        consumed_topic_ids = ts_service.get_consumed_topic_ids(exam_id)
        available_pool = [q for q in pool if q.subtopic.topic_id not in consumed_topic_ids]
        if not available_pool:
            _clear_current_question(exam)
            result = _build_result(exam, answered)
            db.add(exam)
            db.commit()
            db.refresh(exam)
            return _progress_payload(db, exam, result=result)

        next_question = _select_next_question(available_pool, asked_ids, exam.current_theta)
        if not next_question:
            _clear_current_question(exam)
            result = _build_result(exam, answered)
            db.add(exam)
            db.commit()
            db.refresh(exam)
            return _progress_payload(db, exam, result=result)

        _mark_question_served(exam, question_id=next_question.id)
        db.commit()
        db.refresh(exam)
        return _progress_payload(db, exam, next_question=next_question)

    # ── Branch: answered a REGULAR question ────────────────────────
    ts_service.record_answer(
        exam_id, topic_id, theta_after,
        before_order_index=exam.answered_count,
    )

    # If the student switched topics while a lock was active, clear it.
    if exam.locked_topic_id is not None and exam.locked_topic_id != topic_id:
        exam.locked_topic_id = None
        ts_service.clear_generation_attempted(exam_id, topic_id)
        db.add(exam)
        db.flush()

    streak_info = ts_service.get_streak(exam_id, topic_id)

    # ── ISSUE 1 FIX: strict completion guard before any generation/lock routing ──
    if exam.answered_count >= exam.max_questions:
        exam.pending_generated_question_id = None
        _clear_current_question(exam)
        result = _build_result(exam, answered)
        db.add(exam)
        db.commit()
        db.refresh(exam)
        return _progress_payload(db, exam, result=result)

    # ── Currently locked on this topic ─────────────────────────────
    if exam.locked_topic_id == topic_id:
        if streak_info.threshold_reached and not streak_info.generation_attempted:
            next_generated = _try_generate_next(db, exam_id, topic_id, theta_after, student_id)
            ts_service.mark_generation_attempted(exam_id, topic_id)
            if next_generated:
                exam.pending_generated_question_id = next_generated.id
                _mark_question_served(exam, generated_question_id=next_generated.id)
                db.add(exam)
                db.commit()
                db.refresh(exam)
                return _progress_payload(db, exam, next_generated=next_generated)
            # Generation failed — try one more regular Q from same topic
            next_q = _select_next_from_topic(pool, asked_ids, topic_id, theta_after)
            if next_q:
                _mark_question_served(exam, question_id=next_q.id)
                db.commit()
                db.refresh(exam)
                return _progress_payload(db, exam, next_question=next_q)
            # No eligible Qs — unlock and fall through
            exam.locked_topic_id = None
            db.add(exam)
            db.flush()
        elif not streak_info.threshold_reached:
            # Streak still building — stay on this topic
            next_q = _select_next_from_topic(pool, asked_ids, topic_id, theta_after)
            if next_q:
                _mark_question_served(exam, question_id=next_q.id)
                db.commit()
                db.refresh(exam)
                return _progress_payload(db, exam, next_question=next_q)
            # Topic exhausted before threshold — unlock
            exam.locked_topic_id = None
            db.add(exam)
            db.flush()
        else:
            # Threshold already reached and generation already attempted
            # (e.g. generation failed, we served one retry, now moving on)
            ts_service.mark_topic_consumed(exam_id, topic_id)
            exam.locked_topic_id = None
            db.add(exam)
            db.flush()

    # ── Not locked — decide whether to lock or free-select ─────────
    else:
        if not streak_info.threshold_reached and streak_info.current_streak > 0:
            # Start / continue building a streak on this topic
            exam.locked_topic_id = topic_id
            db.add(exam)
            db.flush()
            next_q = _select_next_from_topic(pool, asked_ids, topic_id, theta_after)
            if next_q:
                _mark_question_served(exam, question_id=next_q.id)
                db.commit()
                db.refresh(exam)
                return _progress_payload(db, exam, next_question=next_q)
            # Topic exhausted — unlock and free-select
            exam.locked_topic_id = None
            db.add(exam)
            db.flush()

        elif streak_info.threshold_reached and not streak_info.generation_attempted:
            # Threshold just reached — attempt generation now
            next_generated = _try_generate_next(db, exam_id, topic_id, theta_after, student_id)
            ts_service.mark_generation_attempted(exam_id, topic_id)
            if next_generated:
                exam.locked_topic_id = None
                exam.pending_generated_question_id = next_generated.id
                _mark_question_served(exam, generated_question_id=next_generated.id)
                db.add(exam)
                db.commit()
                db.refresh(exam)
                return _progress_payload(db, exam, next_generated=next_generated)
            # Generation failed — try one more regular Q from same topic
            exam.locked_topic_id = topic_id
            db.add(exam)
            db.flush()
            next_q = _select_next_from_topic(pool, asked_ids, topic_id, theta_after)
            if next_q:
                _mark_question_served(exam, question_id=next_q.id)
                db.commit()
                db.refresh(exam)
                return _progress_payload(db, exam, next_question=next_q)
            # No eligible Qs — unlock
            exam.locked_topic_id = None
            db.add(exam)
            db.flush()

    # ── Free selection across all available topics ──────────────────
    if exam.answered_count >= exam.max_questions:
        _clear_current_question(exam)
        result = _build_result(exam, answered)
        db.add(exam)
        db.commit()
        db.refresh(exam)
        return _progress_payload(db, exam, result=result)

    consumed_topic_ids = ts_service.get_consumed_topic_ids(exam_id)
    available_pool = [q for q in pool if q.subtopic.topic_id not in consumed_topic_ids]
    selection_theta = 0.0 if consumed_topic_ids else exam.current_theta

    if not available_pool:
        _clear_current_question(exam)
        result = _build_result(exam, answered)
        db.add(exam)
        db.commit()
        db.refresh(exam)
        return _progress_payload(db, exam, result=result)

    next_question = _select_next_question(available_pool, asked_ids, selection_theta)
    if not next_question:
        _clear_current_question(exam)
        result = _build_result(exam, answered)
        db.add(exam)
        db.commit()
        db.refresh(exam)
        return _progress_payload(db, exam, result=result)

    _mark_question_served(exam, question_id=next_question.id)
    db.commit()
    db.refresh(exam)
    return _progress_payload(db, exam, next_question=next_question)


def get_adaptive_exam(db: Session, student_id: int, exam_id: int) -> dict:
    exam = db.query(AdaptiveExam).filter(AdaptiveExam.id == exam_id, AdaptiveExam.student_id == student_id).first()
    if not exam:
        raise ValueError("Adaptive exam not found")

    answered = (
        db.query(AdaptiveExamResponse)
        .filter(AdaptiveExamResponse.adaptive_exam_id == exam.id)
        .order_by(AdaptiveExamResponse.order_index.asc())
        .all()
    )

    if exam.status == ExamStatus.Completed:
        result = {
            "score_percent": float(exam.score or 0.0),
            "correct_count": sum(1 for x in answered if x.is_correct),
            "total_questions": len(answered),
            "passed": float(exam.score or 0.0) >= float(get_settings().PHASE2_PASSING_SCORE),
            "final_theta": float(exam.current_theta),
            "message": "Phase II adaptive exam completed",
        }
        return _progress_payload(db, exam, None, result=result)

    # ── ISSUE 2 FIX: restore pending generated question if still unanswered ──
    if exam.pending_generated_question_id is not None:
        pending_gq = (
            db.query(GeneratedQuestion)
            .filter(GeneratedQuestion.id == exam.pending_generated_question_id)
            .first()
        )
        if pending_gq is not None:
            already_answered = (
                db.query(AdaptiveExamResponse)
                .filter(
                    AdaptiveExamResponse.adaptive_exam_id == exam.id,
                    AdaptiveExamResponse.generated_question_id == pending_gq.id,
                )
                .first()
            )
            if already_answered is None:
                # Align the tracked identity with the question we return, but
                # NEVER fabricate a serve timestamp — if the original serve
                # time is missing, the next answer is recorded as untrusted
                # and anomaly scoring is skipped for it.
                if exam.current_generated_question_id != pending_gq.id:
                    exam.current_generated_question_id = pending_gq.id
                    exam.current_question_id = None
                    db.add(exam)
                    db.commit()
                    db.refresh(exam)
                return _progress_payload(db, exam, next_generated=pending_gq, result=None)
            # Already answered — stale pending state, clear it
            exam.pending_generated_question_id = None
            db.add(exam)
            db.commit()
            db.refresh(exam)

    pool = _get_phase_question_pool(db, exam.phase_id, seed_missing_irt=False)
    ts_service = TopicStreakService(db, student_id)
    consumed_topic_ids = ts_service.get_consumed_topic_ids(exam_id)
    available_pool = [q for q in pool if q.subtopic.topic_id not in consumed_topic_ids]
    asked_ids = {r.question_id for r in answered}

    # Resume the previously served, unanswered regular question so timing
    # stays attributable to that exact serve.  Do NOT backfill a missing
    # serve timestamp — a missing timestamp means the next answer is
    # timing-untrusted.
    if exam.current_question_id is not None:
        resume_q = next((q for q in pool if q.id == exam.current_question_id), None)
        if resume_q is not None and resume_q.id not in asked_ids:
            return _progress_payload(db, exam, resume_q, result=None)
        # Tracked question is stale (already answered / no longer in pool) —
        # clear it and fall through to a fresh serve.  Commit so the repair
        # persists; a flush alone is lost when the request session closes.
        exam.current_question_id = None
        exam.current_generated_question_id = None
        exam.current_question_started_at = None
        db.add(exam)
        db.commit()
        db.refresh(exam)

    # If locked to a topic, prefer selecting from that topic
    if exam.locked_topic_id is not None and exam.locked_topic_id not in consumed_topic_ids:
        next_question = _select_next_from_topic(available_pool, asked_ids, exam.locked_topic_id, exam.current_theta)
        if next_question:
            _mark_question_served(exam, question_id=next_question.id)
            db.add(exam)
            db.commit()
            db.refresh(exam)
            return _progress_payload(db, exam, next_question, result=None)

    next_question = _select_next_question(available_pool, asked_ids, exam.current_theta)
    if next_question:
        _mark_question_served(exam, question_id=next_question.id)
        db.add(exam)
        db.commit()
        db.refresh(exam)
        return _progress_payload(db, exam, next_question, result=None)
    return _progress_payload(db, exam, next_question, result=None)
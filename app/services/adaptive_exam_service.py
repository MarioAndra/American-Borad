from __future__ import annotations

import math
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import AdaptiveExam, AdaptiveExamResponse, Choice, ExamStatus, Phase, Question, SubTopic, Topic

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


def _question_payload(db: Session, q: Question) -> dict:
    choices = db.query(Choice).filter(Choice.question_id == q.id).order_by(Choice.id.asc()).all()
    return {
        "id": q.id,
        "text": q.text,
        "difficulty": q.difficulty,
        "cognitive_level": q.cognitive_level,
        "question_type": q.question_type,
        "choices": [{"id": c.id, "text": c.text} for c in choices],
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


def _progress_payload(db: Session, exam: AdaptiveExam, next_question: Question | None, result: dict | None = None) -> dict:
    return {
        "exam_id": exam.id,
        "status": exam.status,
        "phase_id": exam.phase_id,
        "answered_count": exam.answered_count,
        "max_questions": exam.max_questions,
        "remaining_questions": max(exam.max_questions - exam.answered_count, 0),
        "current_theta": float(exam.current_theta),
        "current_question": _question_payload(db, next_question) if next_question else None,
        "result": result,
        "started_at": exam.started_at,
        "submitted_at": exam.submitted_at,
    }


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

    db.commit()
    db.refresh(exam)
    return _progress_payload(db, exam, first_question, result=None)


def submit_adaptive_answer(db: Session, student_id: int, exam_id: int, question_id: int, choice_id: int) -> dict:
    exam = db.query(AdaptiveExam).filter(AdaptiveExam.id == exam_id, AdaptiveExam.student_id == student_id).first()
    if not exam:
        raise ValueError("Adaptive exam not found")
    if exam.status == ExamStatus.Completed:
        raise ValueError("Adaptive exam already completed")

    pool = _get_phase_question_pool(db, exam.phase_id, seed_missing_irt=False)
    by_id = {q.id: q for q in pool}
    q = by_id.get(question_id)
    if not q:
        raise ValueError("Question does not belong to this adaptive exam phase")

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

    theta_before = float(exam.current_theta)
    theta_after = _next_theta(theta_before, is_correct)

    ans = AdaptiveExamResponse(
        adaptive_exam_id=exam.id,
        question_id=question_id,
        choice_id=choice_id,
        order_index=exam.answered_count + 1,
        is_correct=is_correct,
        theta_before=theta_before,
        theta_after=theta_after,
    )
    db.add(ans)

    exam.answered_count += 1
    exam.current_theta = theta_after
    db.add(exam)
    db.flush()

    answered = (
        db.query(AdaptiveExamResponse)
        .filter(AdaptiveExamResponse.adaptive_exam_id == exam.id)
        .order_by(AdaptiveExamResponse.order_index.asc())
        .all()
    )
    asked_ids = {r.question_id for r in answered}

    if exam.answered_count >= exam.max_questions:
        result = _build_result(exam, answered)
        db.add(exam)
        db.commit()
        db.refresh(exam)
        return _progress_payload(db, exam, None, result=result)

    next_question = _select_next_question(pool, asked_ids, exam.current_theta)
    if not next_question:
        result = _build_result(exam, answered)
        db.add(exam)
        db.commit()
        db.refresh(exam)
        return _progress_payload(db, exam, None, result=result)

    db.commit()
    db.refresh(exam)
    return _progress_payload(db, exam, next_question, result=None)


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

    pool = _get_phase_question_pool(db, exam.phase_id, seed_missing_irt=False)
    asked_ids = {r.question_id for r in answered}
    next_question = _select_next_question(pool, asked_ids, exam.current_theta)
    return _progress_payload(db, exam, next_question, result=None)
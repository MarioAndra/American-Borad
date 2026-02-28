from __future__ import annotations
from datetime import datetime, timezone
from typing import Sequence
from sqlalchemy import func, select, case
from sqlalchemy.orm import Session
from app.models import Exam, ExamQuestion, Question, Choice, Phase, Topic, SubTopic
from app.models.enums import DifficultyLevel, ExamStatus


def _passed_phase(db: Session, student_id: int, phase_id: int) -> bool:
    q = (
        db.query(Exam)
        .filter(
            Exam.student_id == student_id,
            Exam.phase_id == phase_id,
            Exam.status == ExamStatus.Completed,
            Exam.score >= 75,
        )
    )
    return db.query(q.exists()).scalar()


def _subq_single_correct() -> select:
    return (
        select(Choice.question_id.label("qid"), func.count(Choice.id).label("cnt"))
        .where(Choice.is_correct == True)  # noqa: E712
        .group_by(Choice.question_id)
        .subquery()
    )


def _random_questions(db: Session, phase_id: int, difficulty: DifficultyLevel, limit: int) -> list[Question]:
    cc = _subq_single_correct()
    q = (
        db.query(Question)
        .join(SubTopic, Question.subtopic_id == SubTopic.id)
        .join(Topic, SubTopic.topic_id == Topic.id)
        .join(Phase, Topic.phase_id == Phase.id)
        .join(cc, cc.c.qid == Question.id)
        .filter(Phase.id == phase_id, Question.difficulty == difficulty, cc.c.cnt == 1, Question.is_active == True)  # noqa: E712
        .order_by(func.random())
        .limit(limit)
    )
    return list(q.all())


def generate_exam(db: Session, student_id: int, phase_id: int) -> Exam:
    if _passed_phase(db, student_id, phase_id):
        raise ValueError("Success! You have already passed Phase. Re-takes are not allowed.")
    easy = _random_questions(db, phase_id, DifficultyLevel.Easy, 33)
    med = _random_questions(db, phase_id, DifficultyLevel.Medium, 34)
    hard = _random_questions(db, phase_id, DifficultyLevel.Hard, 33)
    if len(easy) != 33 or len(med) != 34 or len(hard) != 33:
        raise ValueError("Insufficient questions to satisfy difficulty distribution")
    exam = Exam(
        student_id=student_id,
        phase_id=phase_id,
        total_questions=100,
        easy_count=33,
        medium_count=34,
        hard_count=33,
        status=ExamStatus.Pending,
        score=None,
        started_at=datetime.now(timezone.utc),
        submitted_at=None,
    )
    db.add(exam)
    db.flush()
    picked = [q.id for q in easy + med + hard]
    rows = []
    for i, qid in enumerate(picked, start=1):
        rows.append(ExamQuestion(exam_id=exam.id, question_id=qid, position=i))
    db.add_all(rows)
    db.flush()
    db.commit()
    return exam


def get_exam_with_questions(db: Session, exam_id: int) -> tuple[Exam, list[Question], dict[int, list[Choice]]]:
    exam = db.query(Exam).filter(Exam.id == exam_id).first()
    if not exam:
        raise ValueError("Exam not found")
    qids = [eq.question_id for eq in db.query(ExamQuestion).filter(ExamQuestion.exam_id == exam.id).order_by(ExamQuestion.position)]
    if not qids:
        return exam, [], {}
    qs = list(db.query(Question).filter(Question.id.in_(qids)).all())
    cs = db.query(Choice).filter(Choice.question_id.in_(qids)).all()
    cmap: dict[int, list[Choice]] = {}
    for c in cs:
        cmap.setdefault(c.question_id, []).append(c)
    return exam, qs, cmap


def submit_exam(db: Session, exam_id: int, answers: Sequence[tuple[int, int]]) -> dict:
    exam = db.query(Exam).filter(Exam.id == exam_id).first()
    if not exam:
        raise ValueError("Exam not found")
    if exam.status == ExamStatus.Completed:
        raise ValueError("Exam already submitted")
    eq = db.query(ExamQuestion).filter(ExamQuestion.exam_id == exam.id).all()
    allowed = {x.question_id for x in eq}
    amap = {q: c for q, c in answers if q in allowed}
    correct_map: dict[int, int] = {}
    cs = db.query(Choice).filter(Choice.question_id.in_(list(allowed)), Choice.is_correct == True).all()  # noqa: E712
    for c in cs:
        if c.is_correct:
            correct_map[c.question_id] = c.id
    from app.models import StudentAnswer
    inserted = 0
    correct = 0
    for qid, cid in amap.items():
        is_ok = correct_map.get(qid) == cid
        sa = StudentAnswer(exam_id=exam.id, question_id=qid, choice_id=cid, is_correct=is_ok)
        db.add(sa)
        inserted += 1
        if is_ok:
            correct += 1
    exam.score = round((correct / max(exam.total_questions, 1)) * 100, 2)
    exam.status = ExamStatus.Completed
    exam.submitted_at = datetime.now(timezone.utc)
    db.add(exam)
    db.commit()
    return {"score_percent": float(exam.score or 0), "correct_count": correct, "total_questions": exam.total_questions}

from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.dependencies import get_current_user, get_db, require_roles
from app.models import User, Question, Choice, Exam, ExamQuestion, StudentAnswer
from app.models.enums import ExamStatus
from app.schemas.exam import ExamCreate, ExamRead, ExamSubmission, ExamResult, QuestionRead, ChoiceRead, ExamListItem
from app.services.exam_service import generate_exam, get_exam_with_questions, submit_exam


router = APIRouter()


@router.post("/exams", response_model=ExamRead, dependencies=[Depends(require_roles("Student"))])
def create_exam(payload: ExamCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> ExamRead:
    try:
        exam = generate_exam(db, user.id, payload.phase_id)
        exam_obj, qs, cmap = get_exam_with_questions(db, exam.id)
        qitems = []
        for q in qs:
            ch = [ChoiceRead(id=c.id, text=c.text, is_selected=False) for c in sorted(cmap.get(q.id, []), key=lambda x: x.id)]
            qitems.append(QuestionRead(id=q.id, text=q.text, difficulty=q.difficulty, cognitive_level=q.cognitive_level, question_type=q.question_type, choices=ch))
        return ExamRead(
            id=exam_obj.id,
            phase_id=exam_obj.phase_id,
            total_questions=exam_obj.total_questions,
            easy_count=exam_obj.easy_count,
            medium_count=exam_obj.medium_count,
            hard_count=exam_obj.hard_count,
            status=exam_obj.status,
            score=float(exam_obj.score) if exam_obj.score is not None else None,
            started_at=exam_obj.started_at,
            submitted_at=exam_obj.submitted_at,
            questions=qitems,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/exams/{exam_id}", response_model=ExamRead, dependencies=[Depends(require_roles("Student"))])
def get_exam_detail(exam_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> ExamRead:
    try:
        exam_obj, qs, cmap = get_exam_with_questions(db, exam_id)
        if exam_obj.student_id != user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
        selected_map: dict[int, int] = {}
        if exam_obj.status == ExamStatus.Completed:
            sas = db.query(StudentAnswer).filter(StudentAnswer.exam_id == exam_obj.id).all()
            for sa in sas:
                selected_map[sa.question_id] = sa.choice_id
        qitems = []
        for q in qs:
            choices = sorted(cmap.get(q.id, []), key=lambda x: x.id)
            ch = [ChoiceRead(id=c.id, text=c.text, is_selected=(selected_map.get(q.id) == c.id) if exam_obj.status == ExamStatus.Completed else False) for c in choices]
            qitems.append(QuestionRead(id=q.id, text=q.text, difficulty=q.difficulty, cognitive_level=q.cognitive_level, question_type=q.question_type, choices=ch))
        return ExamRead(
            id=exam_obj.id,
            phase_id=exam_obj.phase_id,
            total_questions=exam_obj.total_questions,
            easy_count=exam_obj.easy_count,
            medium_count=exam_obj.medium_count,
            hard_count=exam_obj.hard_count,
            status=exam_obj.status,
            score=float(exam_obj.score) if exam_obj.score is not None else None,
            started_at=exam_obj.started_at,
            submitted_at=exam_obj.submitted_at,
            questions=qitems,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/exams/{exam_id}/submit", response_model=ExamResult, dependencies=[Depends(require_roles("Student"))])
def submit_exam_answers(exam_id: int, payload: ExamSubmission, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> ExamResult:
    exam = db.query(Exam).filter(Exam.id == exam_id).first()
    if not exam or exam.student_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exam not found")
    ans = [(a.question_id, a.choice_id) for a in payload.answers]
    try:
        result = submit_exam(db, exam_id, ans)
        return ExamResult(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/exams", response_model=list[ExamListItem], dependencies=[Depends(require_roles("Student"))])
def list_my_exams(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[ExamListItem]:
    rows = db.query(Exam).filter(Exam.student_id == user.id).order_by(Exam.created_at.desc()).all()
    out: list[ExamListItem] = []
    for r in rows:
        out.append(ExamListItem(id=r.id, phase_id=r.phase_id, status=r.status, score=float(r.score) if r.score is not None else None, created_at=r.created_at))
    return out

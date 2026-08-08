from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.db.session import get_db
from app.models import (
    Phase,
    Topic,
    SubTopic,
    ABETCriterion,
    Question,
    Choice,
    DifficultyLevel,
    QuestionType,
)
from app.models.enums import cognitive_level_from_value


logger = logging.getLogger("excel_import")


REQUIRED_COLUMNS = [
    "phase",
    "topic",
    "subtopic",
    "cognitive_level",
    "difficulty",
    "question_type",
    "question_text",
    "Option A",
    "Option B",
    "Option C",
    "Option D",
    "correct_answer",
]

OPTION_KEYS = ["Option A", "Option B", "Option C", "Option D"]
LETTER_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


@dataclass
class ImportSummary:
    phases_created: int = 0
    topics_created: int = 0
    subtopics_created: int = 0
    abet_created: int = 0
    questions_created: int = 0
    questions_skipped_duplicates: int = 0
    choices_created: int = 0
    rows_processed: int = 0
    rows_failed: int = 0


def _normalize(s: Optional[str]) -> str:
    return (s or "").strip()


def _map_enum(enum_cls, value: str, field_name: str) -> object:
    v = _normalize(value)
    for member in enum_cls:
        if v.lower() == member.value.lower():
            return member
    raise ValueError(f"Invalid {field_name}: {value}")


class ExcelQuestionImporter:
    def __init__(self, db: Session):
        self.db = db
        self.phases: Dict[str, Phase] = {}
        self.topics: Dict[Tuple[int, str], Topic] = {}
        self.subtopics: Dict[Tuple[int, str], SubTopic] = {}
        self.abet: Dict[str, ABETCriterion] = {}
        self.existing_question_texts: set[str] = set()

    def preload_caches(self) -> None:
        for ph in self.db.query(Phase).all():
            self.phases[_normalize(ph.name)] = ph
        for tp in self.db.query(Topic).all():
            key = (tp.phase_id, _normalize(tp.name))
            self.topics[key] = tp
        for st in self.db.query(SubTopic).all():
            key = (st.topic_id, _normalize(st.name))
            self.subtopics[key] = st
        for ab in self.db.query(ABETCriterion).all():
            self.abet[_normalize(ab.code)] = ab
        for q in self.db.query(Question.text).all():
            self.existing_question_texts.add(_normalize(q[0]))

    def get_or_create_phase(self, name: str, summary: ImportSummary) -> Phase:
        n = _normalize(name)
        ph = self.phases.get(n)
        if ph:
            return ph
        ph = Phase(name=n)
        self.db.add(ph)
        self.db.flush()
        self.phases[n] = ph
        summary.phases_created += 1
        return ph

    def get_or_create_topic(self, phase: Phase, name: str, summary: ImportSummary) -> Topic:
        n = _normalize(name)
        key = (phase.id, n)
        tp = self.topics.get(key)
        if tp:
            return tp
        tp = Topic(phase_id=phase.id, name=n)
        self.db.add(tp)
        self.db.flush()
        self.topics[key] = tp
        summary.topics_created += 1
        return tp

    def get_or_create_subtopic(self, topic: Topic, name: str, summary: ImportSummary) -> SubTopic:
        n = _normalize(name)
        key = (topic.id, n)
        st = self.subtopics.get(key)
        if st:
            return st
        st = SubTopic(topic_id=topic.id, name=n)
        self.db.add(st)
        self.db.flush()
        self.subtopics[key] = st
        summary.subtopics_created += 1
        return st

    def get_or_create_abet(self, code: Optional[str], summary: ImportSummary) -> Optional[ABETCriterion]:
        c = _normalize(code)
        if not c:
            return None
        ab = self.abet.get(c)
        if ab:
            return ab
        ab = ABETCriterion(code=c, name=c, description=None)
        self.db.add(ab)
        self.db.flush()
        self.abet[c] = ab
        summary.abet_created += 1
        return ab

    def import_row(self, row: pd.Series, summary: ImportSummary) -> None:
        summary.rows_processed += 1

        # Validate required fields
        for col in REQUIRED_COLUMNS:
            if _normalize(str(row.get(col, ""))) == "":
                raise ValueError(f"Missing required column: {col}")

        # Prevent duplicate by question_text
        qtext = _normalize(str(row["question_text"]))
        if qtext in self.existing_question_texts:
            summary.questions_skipped_duplicates += 1
            return

        # Enums
        difficulty = _map_enum(DifficultyLevel, str(row["difficulty"]), "difficulty")
        cognitive = cognitive_level_from_value(str(row["cognitive_level"]))
        qtype = _normalize(str(row["question_type"]))

        # Hierarchy
        phase = self.get_or_create_phase(str(row["phase"]), summary)
        topic = self.get_or_create_topic(phase, str(row["topic"]), summary)
        subtopic = self.get_or_create_subtopic(topic, str(row["subtopic"]), summary)

        # ABET by code
        abet = self.get_or_create_abet(row.get("abet_outcomes"), summary)

        # Options and correct answer
        options = [str(row[k]) for k in OPTION_KEYS]
        if any(_normalize(opt) == "" for opt in options):
            raise ValueError("All four options must be provided")
        correct_letter = _normalize(str(row["correct_answer"])).upper()
        if correct_letter not in LETTER_INDEX:
            raise ValueError(f"Invalid correct_answer: {row['correct_answer']}")
        correct_idx = LETTER_INDEX[correct_letter]

        # Question fields
        explanation = _normalize(str(row.get("explanation")))
        common_mistake = _normalize(str(row.get("common_mistake")))
        skill_gap = _normalize(str(row.get("skill_gap")))

        # Create question
        q = Question(
            text=qtext,
            difficulty=difficulty,
            cognitive_level=cognitive,
            question_type=qtype,
            subtopic_id=subtopic.id,
            abet_criterion_id=abet.id if abet else None,
            is_active=True,
            explanation=explanation or None,
            common_mistake=common_mistake or None,
            skill_gap=skill_gap or None,
        )
        self.db.add(q)
        self.db.flush()

        # Create choices
        created_choices = 0
        for i, opt in enumerate(options):
            ch = Choice(
                question_id=q.id,
                text=_normalize(opt),
                is_correct=(i == correct_idx),
            )
            self.db.add(ch)
            created_choices += 1
        self.db.flush()

        summary.questions_created += 1
        summary.choices_created += created_choices
        self.existing_question_texts.add(qtext)


def import_questions_from_excel(db: Session, excel_path: str) -> ImportSummary:
    logger.info("Starting questions import from Excel: %s", excel_path)
    df = pd.read_excel(excel_path, engine="openpyxl")
    def _norm_tokens(s: str) -> list[str]:
        return "".join(ch if ch.isalnum() else " " for ch in s.lower()).split()
    def _map_col(s: str) -> str:
        j = " ".join(_norm_tokens(s))
        if j == "phase":
            return "phase"
        if j == "topic":
            return "topic"
        if j == "subtopic":
            return "subtopic"
        if j == "cognitive level":
            return "cognitive_level"
        if j == "difficulty":
            return "difficulty"
        if j == "question type":
            return "question_type"
        if j == "question text":
            return "question_text"
        if j == "abet outcomes":
            return "abet_outcomes"
        if j == "correct answer" or j == "answer":
            return "correct_answer"
        if j in {"option a", "a"}:
            return "Option A"
        if j in {"option b", "b"}:
            return "Option B"
        if j in {"option c", "c"}:
            return "Option C"
        if j in {"option d", "d"}:
            return "Option D"
        return s.strip()
    df = df.rename(columns={c: _map_col(str(c)) for c in df.columns})
    if any(col not in df.columns for col in REQUIRED_COLUMNS):
        df = pd.read_excel(excel_path, engine="openpyxl", header=1)
        df = df.rename(columns={c: _map_col(str(c)) for c in df.columns})
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Required column missing in Excel: {col}")

    importer = ExcelQuestionImporter(db)
    importer.preload_caches()
    summary = ImportSummary()

    for idx, row in df.iterrows():
        try:
            with db.begin_nested():
                importer.import_row(row, summary)
        except (ValueError, SQLAlchemyError) as e:
            summary.rows_failed += 1
            logger.error("Row %s failed: %s", idx, e)
            continue
    db.commit()

    logger.info(
        "Import finished: phases=%s topics=%s subtopics=%s abet=%s questions=%s duplicates=%s choices=%s failed_rows=%s",
        summary.phases_created,
        summary.topics_created,
        summary.subtopics_created,
        summary.abet_created,
        summary.questions_created,
        summary.questions_skipped_duplicates,
        summary.choices_created,
        summary.rows_failed,
    )
    return summary


def import_questions_cli(excel_path: str) -> ImportSummary:
    # Helper for ad-hoc CLI execution without FastAPI context
    db_gen = get_db()
    db = next(db_gen)
    try:
        return import_questions_from_excel(db, excel_path)
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass

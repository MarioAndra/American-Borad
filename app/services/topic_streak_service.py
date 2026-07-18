from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models import AdaptiveExamResponse, StudentTopicProgress, Topic

log = get_logger(__name__)


@dataclass
class StreakInfo:
    topic_id: int
    topic_name: str
    current_streak: int
    questions_asked: int
    generated_count: int
    avg_theta: float | None
    threshold_reached: bool
    can_generate: bool


class TopicStreakService:
    def __init__(self, db: Session, student_id: int) -> None:
        self.db = db
        self.student_id = student_id
        self.settings = get_settings()
        self._streak_threshold = self.settings.PHASE2_SUBTOPIC_BASE_QUESTION_COUNT
        self._max_generated = self.settings.PHASE2_SUBTOPIC_GENERATED_QUESTION_COUNT

    def record_answer(self, exam_id: int, topic_id: int, theta: float) -> StreakInfo:
        progress = self._get_progress(exam_id, topic_id)

        prev_topic_id = self._previous_topic(exam_id)
        if prev_topic_id is not None and prev_topic_id != topic_id:
            self._reset_topic_progress(exam_id, prev_topic_id)

        progress.current_streak += 1
        progress.questions_asked += 1
        progress.current_theta = theta

        old_total = progress.avg_theta
        if old_total is not None:
            progress.avg_theta = (old_total * (progress.questions_asked - 1) + theta) / progress.questions_asked
        else:
            progress.avg_theta = theta

        self.db.flush()
        return self._build_info(progress)

    def get_streak(self, exam_id: int, topic_id: int) -> StreakInfo:
        progress = self._get_progress(exam_id, topic_id)
        return self._build_info(progress)

    def increment_generated(self, exam_id: int, topic_id: int) -> None:
        progress = self._get_progress(exam_id, topic_id)
        progress.generated_count += 1
        self.db.flush()

    def _get_progress(self, exam_id: int, topic_id: int) -> StudentTopicProgress:
        progress = (
            self.db.query(StudentTopicProgress)
            .filter(
                StudentTopicProgress.exam_id == exam_id,
                StudentTopicProgress.topic_id == topic_id,
            )
            .first()
        )
        if not progress:
            progress = StudentTopicProgress(
                student_id=self.student_id,
                exam_id=exam_id,
                topic_id=topic_id,
                current_streak=0,
                questions_asked=0,
                generated_count=0,
                avg_theta=None,
            )
            self.db.add(progress)
            self.db.flush()
        return progress

    def _previous_topic(self, exam_id: int) -> int | None:
        last = (
            self.db.query(AdaptiveExamResponse)
            .filter(AdaptiveExamResponse.adaptive_exam_id == exam_id)
            .order_by(AdaptiveExamResponse.order_index.desc())
            .first()
        )
        if not last:
            return None
        q = last.question
        return q.subtopic.topic_id if q and q.subtopic else None

    def _reset_topic_progress(self, exam_id: int, topic_id: int) -> None:
        self.db.query(StudentTopicProgress).filter(
            StudentTopicProgress.exam_id == exam_id,
            StudentTopicProgress.topic_id == topic_id,
        ).update({"current_streak": 0})
        self.db.flush()

    def update_topic_theta(self, exam_id: int, topic_id: int, theta: float) -> None:
        progress = self._get_progress(exam_id, topic_id)
        progress.current_theta = theta
        self.db.flush()

    def mark_topic_consumed(self, exam_id: int, topic_id: int) -> None:
        progress = self._get_progress(exam_id, topic_id)
        progress.consumed = True
        self.db.flush()

    def get_consumed_topic_ids(self, exam_id: int) -> set[int]:
        rows = (
            self.db.query(StudentTopicProgress.topic_id)
            .filter(
                StudentTopicProgress.exam_id == exam_id,
                StudentTopicProgress.consumed == True,
            )
            .all()
        )
        return {row[0] for row in rows}

    def is_topic_consumed(self, exam_id: int, topic_id: int) -> bool:
        progress = self._get_progress(exam_id, topic_id)
        return progress.consumed

    def _build_info(self, progress: StudentTopicProgress) -> StreakInfo:
        topic = self.db.query(Topic).filter(Topic.id == progress.topic_id).first()
        topic_name = topic.name if topic else "unknown"
        threshold_reached = progress.current_streak >= self._streak_threshold
        can_generate = threshold_reached and progress.generated_count < self._max_generated

        return StreakInfo(
            topic_id=progress.topic_id,
            topic_name=topic_name,
            current_streak=progress.current_streak,
            questions_asked=progress.questions_asked,
            generated_count=progress.generated_count,
            avg_theta=progress.avg_theta,
            threshold_reached=threshold_reached,
            can_generate=can_generate,
        )

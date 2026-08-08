from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.dependencies import require_roles
from app.db.session import get_db
from app.models import SubTopic, Topic, User
from app.schemas.subtopic import SubTopicRead

router = APIRouter()


@router.get("", response_model=list[SubTopicRead])
def list_subtopics(
    topic_id: Annotated[int | None, Query()] = None,
    admin: User = Depends(require_roles("Admin")),
    db: Session = Depends(get_db),
) -> list[SubTopicRead]:
    query = (
        db.query(SubTopic, Topic)
        .join(Topic, SubTopic.topic_id == Topic.id)
    )
    if topic_id is not None:
        query = query.filter(SubTopic.topic_id == topic_id)
    rows = query.order_by(Topic.name, SubTopic.name).all()

    return [
        SubTopicRead(
            id=subtopic.id,
            topic_id=subtopic.topic_id,
            name=subtopic.name,
            description=subtopic.description,
            topic_name=topic.name,
            created_at=subtopic.created_at,
        )
        for subtopic, topic in rows
    ]

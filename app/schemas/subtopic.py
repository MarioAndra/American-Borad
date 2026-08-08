from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SubTopicRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    topic_id: int
    name: str
    description: str | None = None
    topic_name: str | None = None
    created_at: datetime

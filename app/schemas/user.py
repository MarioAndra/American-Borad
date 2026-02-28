from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, EmailStr, ConfigDict
from app.models.enums import UserRole


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    full_name: str
    email: EmailStr
    role: UserRole
    is_active: bool
    is_verified: bool
    created_at: datetime

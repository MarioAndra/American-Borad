from __future__ import annotations

from typing import Annotated, Literal

from fastapi import Depends, HTTPException, Header, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.core.security import decode_token
from app.models import TokenBlacklist, User, UserRole


def get_authorization_token(authorization: Annotated[str | None, Header()] = None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return authorization.split(" ", 1)[1]


def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(get_authorization_token),
) -> User:
    try:
        payload = decode_token(token)
        user_id = int(payload.get("sub", "0"))
        jti = payload.get("jti")
        ttype = payload.get("type")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    if not jti or ttype != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    revoked = db.query(TokenBlacklist).filter(TokenBlacklist.jti == jti).first()
    if revoked:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token revoked")

    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()  # noqa: E712
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")

    iat = int(payload.get("iat", 0))
    if getattr(user, "password_changed_at", None):
        try:
            if iat < int(user.password_changed_at.timestamp()):
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalidated")
        except Exception:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalidated")
    return user


def require_roles(*roles: Literal["Student", "Admin"]):
    def dependency(user: User = Depends(get_current_user)) -> User:
        if user.role not in {UserRole(role) for role in roles}:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user

    return dependency

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.db.session import get_db
from app.schemas.auth import (
    RegisterRequest,
    LoginRequest,
    TokenPairResponse,
    LogoutResponse,
    RefreshRequest,
    ResendVerificationRequest,
    VerifyCodeRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    DetailResponse,
)
from app.schemas.user import UserRead
from app.services import auth_service

router = APIRouter()


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> UserRead:
    try:
        user = auth_service.register_student(db, payload.full_name, payload.email, payload.password)
        return user  # pydantic from_attributes
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/login", response_model=TokenPairResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenPairResponse:
    try:
        data = auth_service.login(db, payload.email, payload.password)
        return data
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/logout", response_model=LogoutResponse)
def logout(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> LogoutResponse:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    token = authorization.split(" ", 1)[1]
    try:
        auth_service.logout(db, token)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return LogoutResponse()


# Legacy GET verify for email link clicks
@router.get("/verify")
def verify_email(token: str, db: Session = Depends(get_db)) -> dict[str, str]:
    ok = auth_service.verify_email(db, token)
    if not ok:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid verification token")
    return {"detail": "Email verified"}


@router.post("/verify-email", response_model=DetailResponse)
def verify_email_post(payload: VerifyCodeRequest, db: Session = Depends(get_db)) -> DetailResponse:
    ok = auth_service.verify_email_code(db, payload.email, payload.code)
    if not ok:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired code")
    return DetailResponse(detail="Email verified")


@router.post("/resend-verification", response_model=DetailResponse)
def resend_verification(payload: ResendVerificationRequest, db: Session = Depends(get_db)) -> DetailResponse:
    try:
        auth_service.resend_verification(db, payload.email)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e))
    return DetailResponse(detail="If the account exists and is unverified, a new email was sent.")


@router.post("/refresh", response_model=TokenPairResponse)
def refresh(payload: RefreshRequest, db: Session = Depends(get_db)) -> TokenPairResponse:
    try:
        data = auth_service.refresh(db, payload.refresh_token)
        return data
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/logout-all", response_model=LogoutResponse)
def logout_all(
    current_user=Depends(get_current_user),
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> LogoutResponse:
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1]
    auth_service.logout_all(db, current_user.id, token)
    return LogoutResponse()


@router.post("/forgot-password", response_model=DetailResponse)
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)) -> DetailResponse:
    auth_service.forgot_password(db, payload.email)
    return DetailResponse(detail="If the account exists, a password reset email was sent.")


@router.post("/reset-password", response_model=DetailResponse)
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)) -> DetailResponse:
    try:
        auth_service.reset_password(db, payload.token, payload.new_password)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return DetailResponse(detail="Password reset successful")

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    get_password_hash,
    verify_password,
    decode_token,
)
from app.models import TokenBlacklist, User, UserRole
from app.models.refresh_token import RefreshToken
from app.services.mail_service import send_verification_email, send_verification_code_email, send_password_reset_email


def _generate_otp_code(length: int = 6) -> str:
    settings = get_settings()
    n = secrets.randbelow(10**settings.OTP_CODE_LENGTH)
    return f"{n:0{settings.OTP_CODE_LENGTH}d}"[:length]


def _otp_hash(email: str, code: str) -> str:
    settings = get_settings()
    msg = f"{email}:{code}".encode()
    key = settings.JWT_SECRET_KEY.encode()
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def register_student(db: Session, full_name: str, email: str, password: str) -> User:
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise ValueError("Email already registered")
    if not validate_password_policy(password):
        raise ValueError(
            "Password must be at least 8 characters and include uppercase, lowercase, number, and special (!@#$%^&*)."
        )
    now = datetime.now(timezone.utc)
    code = _generate_otp_code()
    code_hash = _otp_hash(email, code)
    user = User(
        full_name=full_name,
        email=email,
        hashed_password=get_password_hash(password),
        role=UserRole.Student,
        is_active=True,
        is_verified=False,
        verification_token=None,
        verification_sent_at=now,
        verification_code_hash=code_hash,
        verification_code_expires_at=now + timedelta(minutes=get_settings().EMAIL_VERIFICATION_CODE_EXPIRE_MINUTES),
        verification_code_attempts=0,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    send_verification_code_email(user.email, code)
    return user


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def login(db: Session, email: str, password: str) -> dict:
    settings = get_settings()
    user = db.query(User).filter(User.email == email, User.is_active == True).first()  # noqa: E712
    if not user:
        raise ValueError("Invalid credentials")

    now = datetime.now(timezone.utc)
    if getattr(user, "locked_until", None) and user.locked_until and now < user.locked_until:
        raise ValueError("Account locked. Try again later.")

    if not verify_password(password, user.hashed_password):
        user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
        user.last_failed_login_at = now
        if user.failed_login_attempts >= settings.FAILED_LOGIN_MAX_ATTEMPTS:
            user.locked_until = now + timedelta(minutes=settings.ACCOUNT_LOCKOUT_MINUTES)
        db.add(user)
        db.commit()
        raise ValueError("Invalid credentials")

    if not user.is_verified:
        raise ValueError("Email not verified")

    user.failed_login_attempts = 0
    user.locked_until = None
    db.add(user)
    db.commit()

    access = create_access_token(str(user.id), user.role.value)
    refresh = create_refresh_token(str(user.id))
    rt = RefreshToken(
        jti=refresh["jti"],
        token_hash=_hash_token(refresh["refresh_token"]),
        user_id=user.id,
        parent_jti=None,
        replaced_by_jti=None,
        revoked=False,
        reason=None,
        expires_at=refresh["expires_at"],
    )
    db.add(rt)
    db.commit()
    return {
        "access_token": access["access_token"],
        "access_expires_at": access["expires_at"],
        "refresh_token": refresh["refresh_token"],
        "refresh_expires_at": refresh["expires_at"],
        "token_type": "bearer",
    }


def logout(db: Session, token: str) -> None:
    payload = decode_token(token)
    jti = payload.get("jti")
    exp = payload.get("exp")
    sub = payload.get("sub")
    ttype = payload.get("type")
    if not jti or not exp or not sub:
        raise ValueError("Invalid token")
    user_id = int(sub)
    if ttype == "access":
        entry = TokenBlacklist(jti=jti, user_id=user_id, expires_at=datetime.fromtimestamp(exp, tz=timezone.utc))
        db.add(entry)
        _revoke_all_user_refresh_tokens(db, user_id, reason="logout")
    elif ttype == "refresh":
        rt = db.query(RefreshToken).filter(RefreshToken.jti == jti, RefreshToken.user_id == user_id).first()
        if rt:
            rt.revoked = True
            rt.reason = "logout"
            db.add(rt)
    db.commit()


def verify_email(db: Session, token: str) -> bool:
    settings = get_settings()
    user = db.query(User).filter(User.verification_token == token).first()
    if not user:
        return False
    if not user.verification_sent_at:
        return False
    expire_at = user.verification_sent_at + timedelta(minutes=settings.EMAIL_VERIFICATION_TOKEN_EXPIRE_MINUTES)
    if datetime.now(timezone.utc) > expire_at:
        return False
    user.is_verified = True
    user.verified_at = datetime.now(timezone.utc)
    user.verification_token = None
    db.add(user)
    db.commit()
    return True


def verify_email_code(db: Session, email: str, code: str) -> bool:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    user = db.query(User).filter(User.email == email, User.is_active == True).first()  # noqa: E712
    if not user or user.is_verified:
        return False
    if user.verification_code_expires_at is None or user.verification_code_hash is None:
        return False
    if now > user.verification_code_expires_at:
        return False
    max_attempts = settings.EMAIL_VERIFICATION_MAX_ATTEMPTS
    if (user.verification_code_attempts or 0) >= max_attempts:
        return False
    expected = _otp_hash(email, code)
    if expected != user.verification_code_hash:
        user.verification_code_attempts = (user.verification_code_attempts or 0) + 1
        db.add(user)
        db.commit()
        return False
    user.is_verified = True
    user.verified_at = now
    user.verification_code_hash = None
    user.verification_code_expires_at = None
    user.verification_code_attempts = 0
    db.add(user)
    db.commit()
    return True


def resend_verification(db: Session, email: str) -> None:
    settings = get_settings()
    user = db.query(User).filter(User.email == email, User.is_active == True).first()  # noqa: E712
    if not user or user.is_verified:
        return
    now = datetime.now(timezone.utc)
    if user.verification_sent_at and (now - user.verification_sent_at) < timedelta(
        minutes=settings.RESEND_VERIFICATION_RATE_LIMIT_MINUTES
    ):
        raise ValueError("Too many requests. Try again later.")
    code = _generate_otp_code()
    user.verification_code_hash = _otp_hash(email, code)
    user.verification_code_expires_at = now + timedelta(minutes=settings.EMAIL_VERIFICATION_CODE_EXPIRE_MINUTES)
    user.verification_code_attempts = 0
    user.verification_token = None
    user.verification_sent_at = now
    db.add(user)
    db.commit()
    send_verification_code_email(user.email, code)


def refresh(db: Session, refresh_token: str) -> dict:
    settings = get_settings()
    payload = decode_token(refresh_token)
    if payload.get("type") != "refresh":
        raise ValueError("Invalid token")
    jti = payload.get("jti")
    sub = payload.get("sub")
    exp = payload.get("exp")
    iat = payload.get("iat", 0)
    if not jti or not sub or not exp:
        raise ValueError("Invalid token")
    user = db.query(User).filter(User.id == int(sub), User.is_active == True).first()  # noqa: E712
    if not user:
        raise ValueError("Invalid token")
    if user.password_changed_at and int(iat) < int(user.password_changed_at.timestamp()):
        raise ValueError("Token invalidated")

    rt = db.query(RefreshToken).filter(RefreshToken.jti == jti, RefreshToken.user_id == user.id).first()
    if not rt:
        raise ValueError("Invalid token")
    if rt.revoked or rt.replaced_by_jti:
        _revoke_all_user_refresh_tokens(db, user.id, reason="refresh token reuse detected")
        raise ValueError("Token reuse detected")
    if datetime.now(timezone.utc) > rt.expires_at:
        raise ValueError("Token expired")
    if rt.token_hash != _hash_token(refresh_token):
        _revoke_all_user_refresh_tokens(db, user.id, reason="refresh token hash mismatch")
        raise ValueError("Token reuse detected")

    rt.revoked = True
    rt.reason = "rotated"
    db.add(rt)

    access = create_access_token(str(user.id), user.role.value)
    new_r = create_refresh_token(str(user.id))
    new_rt = RefreshToken(
        jti=new_r["jti"],
        token_hash=_hash_token(new_r["refresh_token"]),
        user_id=user.id,
        parent_jti=rt.jti,
        replaced_by_jti=None,
        revoked=False,
        reason=None,
        expires_at=new_r["expires_at"],
    )
    rt.replaced_by_jti = new_rt.jti
    db.add(new_rt)
    db.commit()
    return {
        "access_token": access["access_token"],
        "access_expires_at": access["expires_at"],
        "refresh_token": new_r["refresh_token"],
        "refresh_expires_at": new_r["expires_at"],
        "token_type": "bearer",
    }


def logout_all(db: Session, user_id: int, access_token: str | None) -> None:
    if access_token:
        try:
            payload = decode_token(access_token)
            if payload.get("type") == "access":
                jti = payload.get("jti")
                exp = payload.get("exp")
                if jti and exp:
                    entry = TokenBlacklist(
                        jti=jti, user_id=user_id, expires_at=datetime.fromtimestamp(exp, tz=timezone.utc)
                    )
                    db.add(entry)
        except Exception:
            pass
    _revoke_all_user_refresh_tokens(db, user_id, reason="logout_all")
    db.commit()


def _revoke_all_user_refresh_tokens(db: Session, user_id: int, reason: str) -> None:
    q = db.query(RefreshToken).filter(RefreshToken.user_id == user_id, RefreshToken.revoked == False)  # noqa: E712
    for rt in q:
        rt.revoked = True
        rt.reason = reason
        db.add(rt)


def forgot_password(db: Session, email: str) -> None:
    settings = get_settings()
    user = db.query(User).filter(User.email == email, User.is_active == True).first()  # noqa: E712
    if not user:
        return
    token = secrets.token_urlsafe(32)
    user.password_reset_token_hash = _hash_token(token)
    now = datetime.now(timezone.utc)
    user.password_reset_sent_at = now
    user.password_reset_expires_at = now + timedelta(minutes=settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES)
    db.add(user)
    db.commit()
    send_password_reset_email(user.email, token)


def reset_password(db: Session, token: str, new_password: str) -> None:
    if not validate_password_policy(new_password):
        raise ValueError(
            "Password must be at least 8 characters and include uppercase, lowercase, number, and special (!@#$%^&*)."
        )
    now = datetime.now(timezone.utc)
    token_hash = _hash_token(token)
    user = (
        db.query(User)
        .filter(
            User.password_reset_token_hash == token_hash,
            User.password_reset_expires_at != None,  # noqa: E711
            User.password_reset_expires_at > now,
            User.is_active == True,  # noqa: E712
        )
        .first()
    )
    if not user:
        raise ValueError("Invalid or expired token")
    user.hashed_password = get_password_hash(new_password)
    user.password_changed_at = now
    user.password_reset_token_hash = None
    user.password_reset_sent_at = None
    user.password_reset_expires_at = None
    db.add(user)
    _revoke_all_user_refresh_tokens(db, user.id, reason="password_reset")
    db.commit()


def validate_password_policy(password: str) -> bool:
    if not password or len(password) < 8:
        return False
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    special_chars = set("!@#$%^&*")
    has_special = any(c in special_chars for c in password)
    return has_upper and has_lower and has_digit and has_special

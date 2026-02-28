from __future__ import annotations

import hashlib
import bcrypt
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from uuid import uuid4

from jose import JWTError, jwt
from app.core.config import get_settings


_BCRYPT_ROUNDS = 12

def _sha256_prehash(password: str) -> bytes:
    """
    
    
    """
    return hashlib.sha256(password.encode()).hexdigest().encode()


def get_password_hash(password: str) -> str:
    """
    
    """
    password = (password or "").strip()
    prehashed = _sha256_prehash(password)
    salt = bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)
    hashed = bcrypt.hashpw(prehashed, salt)
    return hashed.decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    
    """
    try:
        if not plain_password or not hashed_password:
            return False
        
        plain = plain_password.strip()
        hp_bytes = hashed_password.encode()
        prehashed = _sha256_prehash(plain)
        
        return bcrypt.checkpw(prehashed, hp_bytes)
    except Exception:
        
        return False


def create_access_token(subject: str, role: str) -> dict[str, Any]:
    """
    
    """
    settings = get_settings()
    jti = str(uuid4())
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    payload: Dict[str, Any] = {
        "sub": subject,
        "role": role,
        "type": "access",
        "jti": jti,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return {
        "access_token": token, 
        "token_type": "bearer", 
        "expires_at": expire, 
        "jti": jti
    }


def create_refresh_token(subject: str) -> dict[str, Any]:
    settings = get_settings()
    jti = str(uuid4())
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    payload: Dict[str, Any] = {
        "sub": subject,
        "type": "refresh",
        "jti": jti,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }

    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return {
        "refresh_token": token,
        "expires_at": expire,
        "jti": jti,
    }

def decode_token(token: str) -> dict[str, Any]:
    """
    
    """
    settings = get_settings()
    try:
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET_KEY, 
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError as e:
        raise ValueError("Invalid token") from e

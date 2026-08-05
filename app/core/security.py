"""
Security core: password hashing (bcrypt) + JWT create/decode (python-jose).
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import JWTError, jwt

from app.config import settings


# --- Password Hashing --------------------------------------------------
def hash_password(plain_password: str) -> str:
    """Hash password using bcrypt with 72-byte truncation safety."""
    pwd_bytes = plain_password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against bcrypt hash."""
    pwd_bytes = plain_password.encode('utf-8')[:72]
    hashed_bytes = hashed_password.encode('utf-8')
    try:
        return bcrypt.checkpw(pwd_bytes, hashed_bytes)
    except Exception:
        return False


# --- JWT Access & Refresh Tokens ---------------------------------------
def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create JWT access token containing claims in `data`."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create long-lived JWT refresh token (default 7 days)."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(days=7)
    )
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Decode and validate a JWT access token."""
    return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
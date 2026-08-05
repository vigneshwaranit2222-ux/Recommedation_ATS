from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.database import get_db
from app.models import User, UserRole
from app.schemas import LoginRequest, RefreshTokenRequest, Token, UserCreate, UserResponse
from app.short_id import format_short_id

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

http_bearer = HTTPBearer(auto_error=False)


# --- Registration ----------------------------------------------------------
@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == payload.email))
    existing = result.scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    full_name_val = getattr(payload, "full_name", None) or payload.email.split("@")[0]
    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        full_name=full_name_val,
        role=UserRole(payload.role),
    )
    db.add(user)
    await db.flush()
    return user


# --- Login -------------------------------------------------------------
@router.post("/login", response_model=Token)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Clean JSON login endpoint requiring only email and password."""
    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    claims = {"sub": user.email, "role": user.role.value}
    access_token = create_access_token(data=claims)
    refresh_token = create_refresh_token(data=claims)
    return Token(access_token=access_token, refresh_token=refresh_token)


# --- Refresh Token -----------------------------------------------------
@router.post("/refresh", response_model=Token)
async def refresh_token_endpoint(payload: RefreshTokenRequest, db: AsyncSession = Depends(get_db)):
    try:
        data = decode_access_token(payload.refresh_token)
        if data.get("type") != "refresh":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
        email = data.get("sub")
        if not email:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token claims")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Expired or invalid refresh token")

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    claims = {"sub": user.email, "role": user.role.value}
    new_access_token = create_access_token(data=claims)
    new_refresh_token = create_refresh_token(data=claims)
    return Token(access_token=new_access_token, refresh_token=new_refresh_token)


# --- Dependencies --------------------------------------------------------
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not credentials:
        raise credentials_exception

    token = credentials.credentials
    try:
        payload = decode_access_token(token)
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if user is None:
        raise credentials_exception
    return user


def require_company_role(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != UserRole.company:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Company role required")
    return current_user


def require_candidate_role(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != UserRole.candidate:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Candidate role required")
    return current_user


# --- Current-user info endpoint --------------------------------------------
@router.get("/me", response_model=UserResponse)
async def read_current_user(current_user: User = Depends(get_current_user)):
    return current_user


# --- List users endpoint (for candidate IDs) --------------------------------
@router.get("/users", response_model=list[UserResponse])
async def list_users(db: AsyncSession = Depends(get_db)):
    """List all registered users (candidates, company, admin) with candidate IDs and short_ids (c001, c002)."""
    result = await db.execute(select(User).order_by(User.created_at.asc()))
    users = result.scalars().all()
    resp = []
    for idx, u in enumerate(users, 1):
        u_dict = UserResponse.model_validate(u)
        u_dict.short_id = format_short_id("c", idx)
        resp.append(u_dict)
    return resp
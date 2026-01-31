"""
User API routes - registration, login, and user management.
"""
import hashlib
from datetime import datetime
from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

from app.config import get_settings
from app.database import get_db
from app.models import User


router = APIRouter(prefix="/api/users", tags=["users"])
security = HTTPBearer()

# Simple in-memory token store (in production, use JWT or Redis)
_active_tokens: dict[str, str] = {}  # token -> user_id


# --- Schemas ---

class UserCreate(BaseModel):
    """Schema for user registration."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    anonymous_id: Optional[str] = None


class UserLogin(BaseModel):
    """Schema for user login."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Schema for user response (no sensitive data)."""
    id: str
    email: Optional[str]
    anonymous_id: str
    created_at: str
    last_seen_at: str
    analytics_consent: bool
    available_credits: float
    app_version: Optional[str]


class UserUpdate(BaseModel):
    """Schema for updating user profile."""
    analytics_consent: Optional[bool] = None
    app_version: Optional[str] = None


class TokenResponse(BaseModel):
    """Schema for login response with token."""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# --- Helpers ---

def hash_password(password: str) -> str:
    """Hash password using SHA-256. In production, use bcrypt."""
    # NOTE: In production, use bcrypt or argon2!
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash."""
    return hash_password(password) == password_hash


def generate_token(user_id: str) -> str:
    """Generate and store a token for the user."""
    token = hashlib.sha256(f"{user_id}:{datetime.utcnow()}".encode()).hexdigest()
    _active_tokens[token] = str(user_id)
    return token


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Dependency to get the current authenticated user from Bearer token.
    """
    token = credentials.credentials
    user_id = _active_tokens.get(token)
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


def user_to_response(user: User) -> UserResponse:
    """Convert User model to response schema."""
    return UserResponse(
        id=str(user.id),
        email=user.email,
        anonymous_id=user.anonymous_id,
        created_at=user.created_at.isoformat(),
        last_seen_at=user.last_seen_at.isoformat(),
        analytics_consent=user.analytics_consent,
        available_credits=user.available_credits,
        app_version=user.app_version,
    )


# --- Routes ---

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new user with email and password.
    """
    # Check if email already exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Generate anonymous_id if not provided
    anonymous_id = user_data.anonymous_id or hashlib.sha256(
        user_data.email.encode()
    ).hexdigest()[:32]
    
    # Create user
    user = User(
        email=user_data.email,
        password_hash=hash_password(user_data.password),
        anonymous_id=anonymous_id,
    )
    
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return user_to_response(user)


@router.post("/login", response_model=TokenResponse)
async def login_user(
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db),
):
    """
    Login with email and password.
    
    Returns a Bearer token for authenticated requests.
    """
    result = await db.execute(select(User).where(User.email == credentials.email))
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(credentials.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Update last seen
    user.last_seen_at = datetime.utcnow()
    await db.commit()
    
    # Generate token
    token = generate_token(str(user.id))
    
    return TokenResponse(
        access_token=token,
        user=user_to_response(user)
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
):
    """
    Get the currently authenticated user's profile.
    """
    return user_to_response(current_user)


@router.patch("/me", response_model=UserResponse)
async def update_current_user(
    updates: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update the currently authenticated user's profile.
    
    Users can only update their own profile.
    """
    if updates.analytics_consent is not None:
        current_user.analytics_consent = updates.analytics_consent
    if updates.app_version is not None:
        current_user.app_version = updates.app_version
    
    current_user.last_seen_at = datetime.utcnow()
    await db.commit()
    await db.refresh(current_user)
    
    return user_to_response(current_user)


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_current_user(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete the currently authenticated user's account.
    
    Users can only delete their own account.
    """
    # Invalidate all tokens for this user
    tokens_to_remove = [t for t, uid in _active_tokens.items() if uid == str(current_user.id)]
    for token in tokens_to_remove:
        del _active_tokens[token]
    
    await db.delete(current_user)
    await db.commit()


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """
    Logout and invalidate the current token.
    """
    token = credentials.credentials
    if token in _active_tokens:
        del _active_tokens[token]

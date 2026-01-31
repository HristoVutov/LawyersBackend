"""
User model - User profile for analytics with authentication.
"""
import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, Boolean, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class User(Base):
    """
    User profile for analytics tracking with authentication.
    
    Supports both anonymous tracking and registered accounts.
    """
    __tablename__ = "users"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Anonymous user identifier (hashed machine ID)
    anonymous_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    
    # Authentication fields
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=True)
    
    # When user was first seen
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False
    )
    
    # Last activity timestamp
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False
    )
    
    # Whether user has opted into analytics
    analytics_consent: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # App version (for tracking adoption)
    app_version: Mapped[str] = mapped_column(String(20), nullable=True)
    
    # Billing
    available_credits: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    # Relationships
    transactions = relationship("Transaction", back_populates="user", cascade="all, delete-orphan")


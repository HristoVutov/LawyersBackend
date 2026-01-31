"""
Transaction model - Tracks payments and credit top-ups.
"""
import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Transaction(Base):
    """
    Represents a payment transaction or credit adjustment.
    """
    __tablename__ = "transactions"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    
    # Stripe Details
    stripe_session_id: Mapped[str] = mapped_column(String, unique=True, nullable=True)
    payment_intent_id: Mapped[str] = mapped_column(String, nullable=True)
    
    # Amount Details
    amount_cents: Mapped[int] = mapped_column(Integer, nullable=False) # e.g. 1000 = $10.00
    currency: Mapped[str] = mapped_column(String(3), default="usd")
    credits_amount: Mapped[float] = mapped_column(Float, nullable=False) # e.g. 10.0
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default="pending") # pending, completed, failed
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="transactions")

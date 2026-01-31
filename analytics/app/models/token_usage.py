"""
TokenUsage model - Tracks LLM token consumption per call.
"""
import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, ForeignKey, Integer, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class TokenUsage(Base):
    """
    Records token usage for each LLM call.
    
    Linked to a conversation for detailed cost tracking.
    """
    __tablename__ = "token_usage"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Link to user (for aggregating user-level costs)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    
    # Link to conversation
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )
    
    # Thread ID (fallback if conversation not linked)
    thread_id: Mapped[str] = mapped_column(String(100), nullable=True, index=True)
    
    # Which agent made the call
    agent_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    
    # Model used
    model_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    
    # Token counts
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Cost calculation (optional, in cents or smallest currency unit)
    cost_cents: Mapped[float] = mapped_column(Float, nullable=True)
    
    # When the call was made
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
        index=True
    )
    
    # Additional metadata (response time, etc.)
    # metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

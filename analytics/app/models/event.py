"""
Event model - Analytics event from Lawyers Dashboard.
"""
import uuid
from datetime import datetime
from sqlalchemy import String, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Event(Base):
    """
    Analytics event from a Lawyers Dashboard instance.
    
    Append-only table - events are never updated or deleted.
    """
    __tablename__ = "events"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Anonymous user identifier (hashed, not PII)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    
    # Event type: chat_message, agent_run, tool_call, error, etc.
    event_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    
    # When the event occurred
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        index=True
    )
    
    # Flexible metadata (agent name, model, response time, etc.)
    # metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

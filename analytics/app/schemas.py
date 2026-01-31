"""
Pydantic schemas for API request/response validation.
"""
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class EventCreate(BaseModel):
    """Schema for creating a single event."""
    event_type: str = Field(..., description="Type of event: chat_message, agent_run, tool_call, error")
    user_id: str = Field(..., description="Anonymous user hash (not PII)")
    timestamp: Optional[datetime] = Field(None, description="When event occurred (defaults to now)")
    metadata: Optional[dict] = Field(None, description="Additional event data")


class EventBatch(BaseModel):
    """Schema for batch event submission."""
    events: list[EventCreate] = Field(..., description="List of events to record")


class EventResponse(BaseModel):
    """Response after recording events."""
    received: int = Field(..., description="Number of events received")
    status: str = "ok"


class DailyStatResponse(BaseModel):
    """Daily statistics for a given event type."""
    date: str
    event_type: str
    count: int
    unique_users: int


class StatsResponse(BaseModel):
    """Response containing daily stats."""
    stats: list[DailyStatResponse]
    total_events: int
    period_days: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "0.1.0"

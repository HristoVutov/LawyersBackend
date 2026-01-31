"""
Events API routes - receive and store telemetry events.
"""
from datetime import datetime
from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models import Event, DailyStat
from app.schemas import EventBatch, EventResponse, StatsResponse, DailyStatResponse


router = APIRouter(prefix="/api", tags=["events"])


def verify_api_key(x_api_key: str = Header(None)):
    """Verify the API key from request header."""
    settings = get_settings()
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


@router.post("/events", response_model=EventResponse)
async def receive_events(
    batch: EventBatch,
    db: AsyncSession = Depends(get_db),
    _auth: bool = Depends(verify_api_key),
):
    """
    Receive a batch of telemetry events.
    
    Events are stored asynchronously and never modified.
    """
    events_to_add = []
    
    for event_data in batch.events:
        event = Event(
            user_id=event_data.user_id,
            event_type=event_data.event_type,
            created_at=event_data.timestamp or datetime.utcnow(),
            metadata=event_data.metadata,
        )
        events_to_add.append(event)
    
    db.add_all(events_to_add)
    await db.commit()
    
    return EventResponse(received=len(events_to_add))


@router.get("/stats/daily", response_model=StatsResponse)
async def get_daily_stats(
    days: int = 30,
    event_type: str | None = None,
    db: AsyncSession = Depends(get_db),
    _auth: bool = Depends(verify_api_key),
):
    """
    Get daily aggregated statistics.
    
    Args:
        days: Number of days to look back (default 30)
        event_type: Optional filter by event type
    """
    query = select(DailyStat).order_by(DailyStat.date.desc()).limit(days)
    
    if event_type:
        query = query.where(DailyStat.event_type == event_type)
    
    result = await db.execute(query)
    stats = result.scalars().all()
    
    # Calculate totals
    total_events = sum(s.count for s in stats)
    
    return StatsResponse(
        stats=[
            DailyStatResponse(
                date=str(s.date),
                event_type=s.event_type,
                count=s.count,
                unique_users=s.unique_users,
            )
            for s in stats
        ],
        total_events=total_events,
        period_days=days,
    )

"""
Usage API routes - token usage and conversation tracking.
"""
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from typing import Optional
import uuid

from app.database import get_db
from app.models import User, Conversation, TokenUsage
from app.routes.users import get_current_user


router = APIRouter(prefix="/api/usage", tags=["usage"])


# --- Schemas ---

class TokenUsageCreate(BaseModel):
    """Schema for recording token usage."""
    thread_id: str
    agent_name: str
    model_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_cents: Optional[float] = None
    metadata: Optional[dict] = None


class TokenUsageBatch(BaseModel):
    """Batch of token usage records."""
    records: list[TokenUsageCreate]


class ConversationCreate(BaseModel):
    """Schema for creating/updating a conversation."""
    thread_id: str
    title: Optional[str] = None


class ConversationResponse(BaseModel):
    """Conversation response schema."""
    id: str
    thread_id: str
    title: Optional[str]
    created_at: str
    last_message_at: str
    message_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int


class UsageSummaryResponse(BaseModel):
    """Summary of token usage for a user."""
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_cents: float
    total_conversations: int
    total_messages: int
    period_days: int


class TokenUsageResponse(BaseModel):
    """Single token usage record response."""
    id: str
    agent_name: str
    model_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_cents: Optional[float]
    created_at: str


# --- Routes ---

@router.post("/tokens", status_code=status.HTTP_201_CREATED)
async def record_token_usage(
    batch: TokenUsageBatch,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Record token usage from the client app.
    
    Creates/updates conversations and records each LLM call.
    """
    user_id = current_user.id if current_user else None
    
    for record in batch.records:
        # Find or create conversation
        result = await db.execute(
            select(Conversation).where(
                Conversation.thread_id == record.thread_id,
                Conversation.user_id == user_id
            )
        )
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            conversation = Conversation(
                user_id=user_id,
                thread_id=record.thread_id,
            )
            db.add(conversation)
            await db.flush()  # Get the ID
        
        # Update conversation stats
        conversation.message_count += 1
        conversation.last_message_at = datetime.utcnow()
        conversation.total_input_tokens += record.input_tokens
        conversation.total_output_tokens += record.output_tokens
        
        # Create token usage record
        usage = TokenUsage(
            user_id=user_id,
            conversation_id=conversation.id,
            thread_id=record.thread_id,
            agent_name=record.agent_name,
            model_name=record.model_name,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            total_tokens=record.total_tokens,
            cost_cents=record.cost_cents,
            metadata=record.metadata,
        )
        db.add(usage)
    
    # Calculate total cost to deduct
    total_cost = 0.0
    for record in batch.records:
        if record.cost_cents:
            total_cost += record.cost_cents

    # Deduct from user balance
    if current_user and total_cost > 0:
        # NOTE: This simple logic allows negative balance. 
        # In a real app, you might check if balance > 0 first.
        current_user.available_credits -= total_cost
        db.add(current_user)

    await db.commit()
    
    return {
        "received": len(batch.records), 
        "status": "ok", 
        "remaining_credits": current_user.available_credits if current_user else 0.0
    }


@router.get("/summary", response_model=UsageSummaryResponse)
async def get_usage_summary(
    days: int = 30,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get token usage summary for the current user.
    """
    since = datetime.utcnow() - timedelta(days=days)
    
    # Token usage aggregates
    result = await db.execute(
        select(
            func.coalesce(func.sum(TokenUsage.input_tokens), 0).label("input"),
            func.coalesce(func.sum(TokenUsage.output_tokens), 0).label("output"),
            func.coalesce(func.sum(TokenUsage.total_tokens), 0).label("total"),
            func.coalesce(func.sum(TokenUsage.cost_cents), 0.0).label("cost"),
        ).where(
            TokenUsage.user_id == current_user.id,
            TokenUsage.created_at >= since
        )
    )
    row = result.one()
    
    # Conversation count
    conv_result = await db.execute(
        select(func.count(Conversation.id)).where(
            Conversation.user_id == current_user.id,
            Conversation.created_at >= since
        )
    )
    conv_count = conv_result.scalar() or 0
    
    # Message count
    msg_result = await db.execute(
        select(func.coalesce(func.sum(Conversation.message_count), 0)).where(
            Conversation.user_id == current_user.id,
            Conversation.created_at >= since
        )
    )
    msg_count = msg_result.scalar() or 0
    
    return UsageSummaryResponse(
        total_input_tokens=row.input,
        total_output_tokens=row.output,
        total_tokens=row.total,
        total_cost_cents=row.cost,
        total_conversations=conv_count,
        total_messages=msg_count,
        period_days=days,
    )


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List user's conversations with usage stats.
    """
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.last_message_at.desc())
        .limit(limit)
        .offset(offset)
    )
    conversations = result.scalars().all()
    
    return [
        ConversationResponse(
            id=str(c.id),
            thread_id=c.thread_id,
            title=c.title,
            created_at=c.created_at.isoformat(),
            last_message_at=c.last_message_at.isoformat(),
            message_count=c.message_count,
            total_input_tokens=c.total_input_tokens,
            total_output_tokens=c.total_output_tokens,
            total_tokens=c.total_input_tokens + c.total_output_tokens,
        )
        for c in conversations
    ]


@router.get("/conversations/{conversation_id}/tokens", response_model=list[TokenUsageResponse])
async def get_conversation_token_usage(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get detailed token usage for a specific conversation.
    """
    # Verify conversation belongs to user
    conv_result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        )
    )
    if not conv_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    result = await db.execute(
        select(TokenUsage)
        .where(TokenUsage.conversation_id == conversation_id)
        .order_by(TokenUsage.created_at.asc())
    )
    records = result.scalars().all()
    
    return [
        TokenUsageResponse(
            id=str(r.id),
            agent_name=r.agent_name,
            model_name=r.model_name,
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
            total_tokens=r.total_tokens,
            cost_cents=r.cost_cents,
            created_at=r.created_at.isoformat(),
        )
        for r in records
    ]


# --- Time-based breakdown ---

class TimePeriodUsage(BaseModel):
    """Usage for a single time period."""
    period: str  # ISO format date/datetime
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_cents: float
    call_count: int


class UsageBreakdownResponse(BaseModel):
    """Usage broken down by time period."""
    period_type: str  # hour, day, week, month
    periods: list[TimePeriodUsage]
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_cents: float


@router.get("/breakdown", response_model=UsageBreakdownResponse)
async def get_usage_breakdown(
    period: str = "day",  # hour, day, week, month
    count: int = 30,  # number of periods to return
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get token usage broken down by time period.
    
    Args:
        period: One of 'hour', 'day', 'week', 'month'
        count: Number of periods to return (default 30)
    """
    # Calculate date range based on period
    now = datetime.utcnow()
    
    if period == "hour":
        since = now - timedelta(hours=count)
        trunc_format = "hour"
    elif period == "day":
        since = now - timedelta(days=count)
        trunc_format = "day"
    elif period == "week":
        since = now - timedelta(weeks=count)
        trunc_format = "week"
    elif period == "month":
        since = now - timedelta(days=count * 30)  # Approximate
        trunc_format = "month"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="period must be one of: hour, day, week, month"
        )
    
    # PostgreSQL date_trunc for grouping
    period_col = func.date_trunc(trunc_format, TokenUsage.created_at)
    
    result = await db.execute(
        select(
            period_col.label("period"),
            func.coalesce(func.sum(TokenUsage.input_tokens), 0).label("input_tokens"),
            func.coalesce(func.sum(TokenUsage.output_tokens), 0).label("output_tokens"),
            func.coalesce(func.sum(TokenUsage.total_tokens), 0).label("total_tokens"),
            func.coalesce(func.sum(TokenUsage.cost_cents), 0.0).label("cost_cents"),
            func.count(TokenUsage.id).label("call_count"),
        )
        .where(
            TokenUsage.user_id == current_user.id,
            TokenUsage.created_at >= since
        )
        .group_by(period_col)
        .order_by(period_col.desc())
        .limit(count)
    )
    
    rows = result.all()
    
    periods = [
        TimePeriodUsage(
            period=row.period.isoformat() if row.period else "",
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            total_tokens=row.total_tokens,
            cost_cents=row.cost_cents,
            call_count=row.call_count,
        )
        for row in rows
    ]
    
    # Calculate totals
    total_input = sum(p.input_tokens for p in periods)
    total_output = sum(p.output_tokens for p in periods)
    total_cost = sum(p.cost_cents for p in periods)
    
    return UsageBreakdownResponse(
        period_type=period,
        periods=periods,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=total_input + total_output,
        total_cost_cents=total_cost,
    )


@router.get("/by-agent", response_model=list[dict])
async def get_usage_by_agent(
    days: int = 30,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get token usage grouped by agent.
    """
    since = datetime.utcnow() - timedelta(days=days)
    
    result = await db.execute(
        select(
            TokenUsage.agent_name,
            func.coalesce(func.sum(TokenUsage.input_tokens), 0).label("input_tokens"),
            func.coalesce(func.sum(TokenUsage.output_tokens), 0).label("output_tokens"),
            func.coalesce(func.sum(TokenUsage.total_tokens), 0).label("total_tokens"),
            func.coalesce(func.sum(TokenUsage.cost_cents), 0.0).label("cost_cents"),
            func.count(TokenUsage.id).label("call_count"),
        )
        .where(
            TokenUsage.user_id == current_user.id,
            TokenUsage.created_at >= since
        )
        .group_by(TokenUsage.agent_name)
        .order_by(func.sum(TokenUsage.total_tokens).desc())
    )
    
    return [
        {
            "agent_name": row.agent_name,
            "input_tokens": row.input_tokens,
            "output_tokens": row.output_tokens,
            "total_tokens": row.total_tokens,
            "cost_cents": row.cost_cents,
            "call_count": row.call_count,
        }
        for row in result.all()
    ]


@router.get("/by-model", response_model=list[dict])
async def get_usage_by_model(
    days: int = 30,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get token usage grouped by model.
    """
    since = datetime.utcnow() - timedelta(days=days)
    
    result = await db.execute(
        select(
            TokenUsage.model_name,
            func.coalesce(func.sum(TokenUsage.input_tokens), 0).label("input_tokens"),
            func.coalesce(func.sum(TokenUsage.output_tokens), 0).label("output_tokens"),
            func.coalesce(func.sum(TokenUsage.total_tokens), 0).label("total_tokens"),
            func.coalesce(func.sum(TokenUsage.cost_cents), 0.0).label("cost_cents"),
            func.count(TokenUsage.id).label("call_count"),
        )
        .where(
            TokenUsage.user_id == current_user.id,
            TokenUsage.created_at >= since
        )
        .group_by(TokenUsage.model_name)
        .order_by(func.sum(TokenUsage.total_tokens).desc())
    )
    
    return [
        {
            "model_name": row.model_name,
            "input_tokens": row.input_tokens,
            "output_tokens": row.output_tokens,
            "total_tokens": row.total_tokens,
            "cost_cents": row.cost_cents,
            "call_count": row.call_count,
        }
        for row in result.all()
    ]


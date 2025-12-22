"""
Feedback API - Endpoint for submitting user feedback on agent responses.

Allows frontend to submit user satisfaction scores that get attached
to LangSmith traces for analysis and improvement.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

from app.tracing import add_feedback, is_tracing_enabled

router = APIRouter()


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""
    
    run_id: str = Field(
        ...,
        description="LangSmith run ID to attach feedback to",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Feedback score from 0.0 (negative) to 1.0 (positive)",
        examples=[1.0, 0.0, 0.75]
    )
    feedback_type: Literal["user_rating", "accuracy", "helpfulness", "relevance"] = Field(
        default="user_rating",
        description="Type of feedback being submitted"
    )
    comment: str | None = Field(
        default=None,
        description="Optional text comment from user",
        examples=["Very helpful response!"]
    )


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    
    status: Literal["ok", "error"]
    message: str
    tracing_enabled: bool


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for an agent response.
    
    This attaches feedback to the LangSmith trace, allowing you to:
    - Track user satisfaction over time
    - Identify problematic responses for improvement
    - Create datasets from highly-rated or poorly-rated interactions
    
    **Feedback types:**
    - `user_rating`: General thumbs up/down (1.0 or 0.0)
    - `accuracy`: How accurate was the response
    - `helpfulness`: How helpful was the response
    - `relevance`: How relevant was the response to the query
    """
    if not is_tracing_enabled():
        return FeedbackResponse(
            status="ok",
            message="Feedback noted (LangSmith tracing not enabled)",
            tracing_enabled=False
        )
    
    success = await add_feedback(
        run_id=request.run_id,
        score=request.score,
        feedback_key=request.feedback_type,
        comment=request.comment,
        source_info={"source": "api", "type": request.feedback_type}
    )
    
    if success:
        return FeedbackResponse(
            status="ok",
            message=f"Feedback submitted successfully for run {request.run_id}",
            tracing_enabled=True
        )
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to submit feedback to LangSmith"
        )


@router.get("/feedback/status")
async def feedback_status():
    """
    Check if feedback/tracing is enabled.
    
    Returns the current tracing status so frontend can conditionally
    show feedback UI elements.
    """
    from app.tracing import get_langsmith_url
    
    return {
        "tracing_enabled": is_tracing_enabled(),
        "langsmith_url": get_langsmith_url(),
    }

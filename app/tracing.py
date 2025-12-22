"""
LangSmith Tracing Utilities.

Provides helpers for tracing agent execution with LangSmith.
Requires LANGCHAIN_API_KEY to be set in environment.

Key concepts from LangSmith:
- Runs: Individual execution units (LLM calls, tool invocations)
- Traces: Collection of runs forming complete execution flow
- Threads: Group related traces (e.g., multi-turn conversations)
- Projects: Organizational containers for traces
- Feedback: Quality signals attached to traces
- Tags: Labels for categorizing and filtering runs
- Metadata: Key-value pairs for custom context
"""
import os
from functools import wraps
from typing import Callable, Any
from contextlib import contextmanager
from datetime import datetime

from app.config import get_settings


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is enabled."""
    settings = get_settings()
    return settings.langsmith_enabled


def setup_tracing():
    """Configure LangSmith tracing environment variables."""
    settings = get_settings()
    
    if settings.langsmith_enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        print(f"📊 LangSmith tracing enabled for project: {settings.langchain_project}")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"


@contextmanager
def trace_run(name: str, run_type: str = "chain", metadata: dict | None = None):
    """
    Context manager for tracing a run in LangSmith.
    
    Usage:
        with trace_run("search_documents", run_type="tool"):
            result = search_indexed(query)
    """
    if not is_tracing_enabled():
        yield
        return
    
    try:
        from langsmith import trace
        
        with trace(name=name, run_type=run_type, metadata=metadata or {}):
            yield
    except ImportError:
        # LangSmith not installed, skip tracing
        yield
    except Exception as e:
        print(f"⚠️ Tracing error: {e}")
        yield


def traced(name: str | None = None, run_type: str = "chain"):
    """
    Decorator to trace a function call in LangSmith.
    
    Usage:
        @traced("agent_invoke")
        async def invoke_agent(message: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        trace_name = name or func.__name__
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            with trace_run(trace_name, run_type):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            with trace_run(trace_name, run_type):
                return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if hasattr(func, "__wrapped__"):
            return async_wrapper
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def get_langsmith_url() -> str | None:
    """Get the LangSmith project URL if tracing is enabled."""
    settings = get_settings()
    if settings.langsmith_enabled:
        return f"https://smith.langchain.com/project/{settings.langchain_project}"
    return None


def get_langsmith_client():
    """
    Get a LangSmith client instance for direct API access.
    
    Returns None if tracing is not enabled.
    """
    if not is_tracing_enabled():
        return None
    
    try:
        from langsmith import Client
        return Client()
    except ImportError:
        print("⚠️ langsmith package not installed")
        return None
    except Exception as e:
        print(f"⚠️ Failed to create LangSmith client: {e}")
        return None


def get_run_callbacks(
    thread_id: str,
    agent_name: str = "orchestrator",
    user_id: str | None = None,
    session_id: str | None = None,
    extra_tags: list[str] | None = None,
    extra_metadata: dict | None = None,
) -> list:
    """
    Create LangChain callbacks with proper tracing context.
    
    This configures traces with:
    - Thread ID for grouping conversation turns
    - Agent name as a tag for filtering
    - Custom metadata for context
    
    Args:
        thread_id: Unique identifier for the conversation thread
        agent_name: Name of the agent (used as tag)
        user_id: Optional user identifier
        session_id: Optional session identifier (alternative to thread_id)
        extra_tags: Additional tags to attach
        extra_metadata: Additional metadata key-value pairs
        
    Returns:
        List of callback handlers to pass to LangChain/LangGraph
    """
    if not is_tracing_enabled():
        return []
    
    try:
        from langchain_core.tracers import LangChainTracer
        
        settings = get_settings()
        
        # Build tags list
        tags = [agent_name, "lawyers-dashboard"]
        if extra_tags:
            tags.extend(extra_tags)
        
        # Build metadata dict
        metadata = {
            "thread_id": thread_id,
            "agent": agent_name,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if user_id:
            metadata["user_id"] = user_id
        if session_id:
            metadata["session_id"] = session_id
        if extra_metadata:
            metadata.update(extra_metadata)
        
        tracer = LangChainTracer(
            project_name=settings.langchain_project,
            tags=tags,
        )
        
        # Note: LangChainTracer doesn't directly accept metadata in constructor
        # Metadata is typically passed via RunnableConfig. We return the tracer
        # and the caller should set metadata in the config dict.
        
        return [tracer], metadata
        
    except ImportError:
        print("⚠️ langchain_core.tracers not available")
        return [], {}
    except Exception as e:
        print(f"⚠️ Failed to create run callbacks: {e}")
        return [], {}


async def add_feedback(
    run_id: str,
    score: float,
    feedback_key: str = "user_rating",
    comment: str | None = None,
    source_info: dict | None = None,
) -> bool:
    """
    Attach user feedback to a trace run.
    
    Feedback can be:
    - Thumbs up/down (score: 1.0 or 0.0)
    - Rating scale (score: 0.0 to 1.0)
    - With optional comment
    
    Args:
        run_id: The LangSmith run ID to attach feedback to
        score: Feedback score from 0.0 (negative) to 1.0 (positive)
        feedback_key: Type of feedback (e.g., "user_rating", "accuracy", "helpfulness")
        comment: Optional text comment from user
        source_info: Optional dict with feedback source context
        
    Returns:
        True if feedback was successfully submitted, False otherwise
    """
    if not is_tracing_enabled():
        print("📊 LangSmith tracing not enabled, feedback not submitted")
        return False
    
    client = get_langsmith_client()
    if not client:
        return False
    
    try:
        client.create_feedback(
            run_id=run_id,
            key=feedback_key,
            score=score,
            comment=comment,
            source_info=source_info,
        )
        print(f"📊 Feedback submitted for run {run_id}: {feedback_key}={score}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to submit feedback: {e}")
        return False


def get_current_run_id() -> str | None:
    """
    Get the current LangSmith run ID from context.
    
    This can be used to retrieve the run ID for feedback submission.
    Note: This only works within an active traced context.
    """
    try:
        from langsmith import get_current_run_tree
        
        run_tree = get_current_run_tree()
        if run_tree:
            return str(run_tree.id)
    except ImportError:
        pass
    except Exception:
        pass
    
    return None

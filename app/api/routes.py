"""
HTTP API Routes.

Provides REST endpoints for health checks, session management, and non-streaming requests.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path

from app.config import get_settings
from app.middleware.virtual_fs import get_vfs, set_project_path

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    google_api_configured: bool
    langsmith_enabled: bool
    indexed_files_dir: str


class SessionState(BaseModel):
    """Session state for a conversation."""
    conversation_id: str
    documents_in_context: list[str] = []
    documents_read: list[str] = []
    entities: dict = {}


class ProjectPathRequest(BaseModel):
    """Request to set active project path."""
    project_path: str


class ProjectPathResponse(BaseModel):
    """Response after setting project path."""
    status: str
    project_path: str
    mounts: dict


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the current status of the backend including:
    - Whether API keys are configured
    - LangSmith tracing status
    - Indexed files directory path
    """
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="0.1.0",
        google_api_configured=bool(settings.google_api_key),
        langsmith_enabled=settings.langsmith_enabled,
        indexed_files_dir=str(settings.indexed_files_dir),
    )


@router.post("/project", response_model=ProjectPathResponse)
async def set_project(request: ProjectPathRequest):
    """
    Set the active project path for file operations.
    
    This configures the virtual file system mounts:
    - /project/ -> project_path (read-only)
    - /output/ -> project_path/.generatedFiles (read/write)
    """
    project_path = request.project_path
    
    # Validate path exists
    if not Path(project_path).exists():
        raise HTTPException(status_code=400, detail=f"Project path does not exist: {project_path}")
    
    # Set the project in virtual filesystem
    set_project_path(project_path)
    
    # Get current mounts for response
    vfs = get_vfs()
    mounts = vfs.list_mounts()
    
    return ProjectPathResponse(
        status="success",
        project_path=project_path,
        mounts=mounts,
    )


@router.get("/project/mounts")
async def get_mounts():
    """
    Get current virtual filesystem mount points.
    
    Returns all configured virtual paths and their real locations.
    """
    vfs = get_vfs()
    return {
        "mounts": vfs.list_mounts(),
        "project_configured": vfs._project_path is not None,
    }


@router.get("/session/{conversation_id}", response_model=SessionState)
async def get_session(conversation_id: str):
    """
    Get session state for a conversation.
    
    Returns documents in context, documents read, and remembered entities.
    """
    # TODO: Implement with session store in Phase 2
    return SessionState(
        conversation_id=conversation_id,
        documents_in_context=[],
        documents_read=[],
        entities={},
    )


@router.delete("/session/{conversation_id}")
async def clear_session(conversation_id: str):
    """
    Clear session state for a conversation.
    
    Removes all cached documents and entities.
    """
    # TODO: Implement with session store in Phase 2
    return {"status": "cleared", "conversation_id": conversation_id}

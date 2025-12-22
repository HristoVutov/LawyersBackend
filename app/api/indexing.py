from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from app.services.indexer import index_directory, get_indexing_status
from app.config import get_settings
from app.middleware.virtual_fs import get_vfs

router = APIRouter()

class IndexRequest(BaseModel):
    path: str


class IndexingStatusResponse(BaseModel):
    """Response for indexing status."""
    total_files: int
    indexed_files: int
    stale_files: int
    status: str  # empty, not_indexed, partially_indexed, fully_indexed


@router.post("/index/start")
async def start_indexing(request: IndexRequest, background_tasks: BackgroundTasks):
    """Start background indexing of a directory."""
    from app.api.routes import set_project_path
    
    # 1. Set the project path so indexes go to the right place (.indexedfiles)
    # This also mounts the directory as /project/
    try:
        set_project_path(request.path)
    except Exception as e:
        return {"status": "error", "message": f"Invalid project path: {str(e)}"}

    # 2. Start indexing in background
    background_tasks.add_task(index_directory, request.path)
    return {"status": "started", "path": request.path, "project_set": True}


@router.get("/index/status", response_model=IndexingStatusResponse)
async def get_project_indexing_status():
    """
    Get the indexing status for the current project.
    
    Returns counts of total files, indexed files, and stale files,
    plus an overall status: empty, not_indexed, partially_indexed, fully_indexed.
    """
    vfs = get_vfs()
    
    if not vfs._project_path:
        raise HTTPException(status_code=400, detail="No active project set")
    
    result = get_indexing_status(vfs._project_path)
    return IndexingStatusResponse(**result)


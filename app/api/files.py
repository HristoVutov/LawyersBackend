"""
API Router for File Operations.

Handles document splitting and other file-specific actions.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path

from app.services.indexer import create_virtual_index
from app.agents.schemas.analysis_schemas import DocumentSegment
from app.config import get_settings
from app.middleware.virtual_fs import get_vfs
import json
import aiofiles
import asyncio

router = APIRouter()

class SplitRequest(BaseModel):
    file_path: str
    segments: Optional[List[DocumentSegment]] = None

@router.post("/files/split")
async def split_document(request: SplitRequest, background_tasks: BackgroundTasks):
    """
    Split a document into virtual segments based on provided boundaries.
    
    This does NOT create new physical files, but creates new index files
    (.index.json) that allow the system to treat segments as distinct docs.
    """
    vfs = get_vfs()
    if not vfs._project_path:
        raise HTTPException(status_code=400, detail="No active project set")
        
    # Validate file exists in index
    # Validate file exists in index
    settings = get_settings()
    
    # Handle both raw filenames ("Contract.pdf") and index filenames ("Contract.pdf.index.json")
    if request.file_path.endswith(".index.json"):
        index_filename = request.file_path
        logical_file_path = request.file_path.replace(".index.json", "")
    else:
        index_filename = f"{request.file_path}.index.json"
        logical_file_path = request.file_path
        
    index_file = settings.indexed_files_dir / index_filename
    
    if not index_file.exists():
         raise HTTPException(status_code=404, detail="File not found in index. Must be indexed first.")

    created_files = []
    
    segment_list = request.segments
    
    # If no segments provided, load from index
    if not segment_list:
        try:
             async with aiofiles.open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.loads(await f.read())
                
             raw_segments = index_data.get("documentSegments", [])
             if not raw_segments:
                 raise HTTPException(status_code=400, detail="No segments found in index file.")
                 
             # Validate/Convert to Pydantic models
             segment_list = [DocumentSegment(**s) for s in raw_segments]
             
        except Exception as e:
             raise HTTPException(status_code=500, detail=f"Failed to load segments from index: {e}")

    # Process segments
    # Running in parallel with a semaphore to control concurrency
    # This prevents timeouts on large documents (e.g. 20+ segments)
    semaphore = asyncio.Semaphore(5)

    async def process_segment(i, segment):
        async with semaphore:
            return await create_virtual_index(logical_file_path, i, segment)

    try:
        tasks = [process_segment(i, segment) for i, segment in enumerate(segment_list, 1)]
        created_files = await asyncio.gather(*tasks)
             
        # Mark parent as splitted
        async with aiofiles.open(index_file, 'r', encoding='utf-8') as f:
            final_index_data = json.loads(await f.read())
            
        final_index_data["splitted"] = True
        
        async with aiofiles.open(index_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(final_index_data, ensure_ascii=False, indent=2))
             
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Splitting failed: {str(e)}")
        
    return {
        "status": "success",
        "message": f"Created {len(created_files)} virtual segments",
        "segments": created_files
    }

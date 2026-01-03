
import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.api.files import split_document, SplitRequest
from app.agents.schemas.analysis_schemas import DocumentSegment

@pytest.mark.asyncio
async def test_split_concurrency_mock():
    """
    Verify that segments are processed in parallel.
    We mock create_virtual_index to take 1 second.
    If 5 segments take ~1 second total, it's parallel.
    If it takes ~5 seconds, it's sequential.
    """
    
    # Mock settings to avoid file system errors
    with patch("app.api.files.get_settings") as mock_settings, \
         patch("app.api.files.get_vfs") as mock_vfs, \
         patch("app.api.files.create_virtual_index", new_callable=AsyncMock) as mock_create_index, \
         patch("app.api.files.aiofiles.open") as mock_open: # Default is MagicMock
        
        # Setup mocks
        mock_vfs.return_value._project_path = "C:/fake/project"
        mock_settings.return_value.indexed_files_dir.exists.return_value = True
        mock_settings.return_value.indexed_files_dir.exists.return_value = True
        
        async def delayed_create(*args, **kwargs):
            await asyncio.sleep(0.5)
            return "path/to/virtual_index"
            
        mock_create_index.side_effect = delayed_create
        
        # Mock index file reading/writing
        mock_file_handle = AsyncMock()
        mock_file_handle.read.return_value = '{"documentSegments": []}'
        mock_open.return_value.__aenter__.return_value = mock_file_handle
        
        # Prepare request with 5 segments
        segments = [
            DocumentSegment(summary=f"Seg {i}", segmentType="Clause", startText="start", endText="end")
            for i in range(5)
        ]
        
        request = SplitRequest(
            file_path="Contract.pdf",
            segments=segments
        )
        
        print("Starting parallel split test...")
        start_time = asyncio.get_event_loop().time()
        
        # Call the function directly (since it's async)
        # We need to mock BackgroundTasks but we can pass None if not used, 
        # but the signature requires it.
        from fastapi import BackgroundTasks
        await split_document(request, BackgroundTasks())
        
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        
        print(f"Total duration: {duration:.2f}s")
        
        # With 5 segments * 0.5s:
        # Sequential ~ 2.5s
        # Parallel ~ 0.5s (plus overhead)
        # We check if it's under 1.0s
        assert duration < 1.0, f"Processing took {duration:.2f}s, expected < 1.0s for parallel execution"
        print("Test passed: Parallel execution verified.")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(test_split_concurrency_mock())

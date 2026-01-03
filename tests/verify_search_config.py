
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure the app package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock settings
mock_settings = MagicMock()
mock_settings.search_result_default_number = 3
mock_settings.google_api_key = "fake_key"
mock_settings.indexed_files_dir = Path("./mock_index_dir")

# Mock get_settings
with patch("app.config.get_settings", return_value=mock_settings):
    from app.tools.search_tools import search_indexed

async def test_search_default_limit():
    print("🧪 Testing Search Default Limit Configuration")
    
    # Mock dependencies to avoid actual FS/API calls
    with patch("app.tools.search_tools.get_cached_search", return_value=None), \
         patch("app.tools.search_tools.find_entity", return_value=None), \
         patch("app.tools.search_tools.get_vfs") as mock_vfs, \
         patch("app.tools.search_tools.get_query_embedding", return_value=[0.1]*768), \
         patch("pathlib.Path.read_text", return_value='{"content": "Match", "summary": "Summary", "embedding": [0.1]*768}'), \
         patch("app.tools.search_tools.cache_search_results"), \
         patch("app.tools.search_tools.extract_entities_from_results", return_value=[]):
        
        # Setup mock VFS
        mock_vfs.return_value._project_path = Path("/tmp/mock_project")
        
        # Setup mock index files
        mock_file = MagicMock()
        mock_file.name = "doc1.index.json"
        mock_file.exists.return_value = True
        mock_file.read_text.return_value = '{"content": "test query match", "summary": "A test document", "keywords": ["test"], "embedding": [0.1]*768}'
        
        # Make iterdir return enough files to test the limit
        # We want to verify that if we have 10 files, only 3 are returned (our mock default)
        files = []
        for i in range(10):
            f = MagicMock()
            f.name = f"doc{i}.index.json"
            f.read_text.return_value = '{"content": "test query match", "summary": "A test document", "keywords": ["test"], "embedding": [0.1]*768}'
            files.append(f)
            
        # Mock the directory iteration
        # Note: In the actual code, it iterates over search_dirs. 
        # We need to mock path iteration.
        
        with patch("pathlib.Path.iterdir", return_value=files):
             # Execute search WITHOUT limit argument
            print("   Calling search_indexed(query='test') without limit...")
            result = await search_indexed.invoke({"query": "test"})
            
            # Count results
            count = result.count("📄")
            print(f"   Results found: {count}")
            
            if count == 3:
                print("✅ SUCCESS: Returned exactly 3 results (matching mock default)")
            else:
                print(f"❌ FAILURE: Returned {count} results (expected 3)")
                print("   Note: Ensure the code uses settings.search_result_default_number when limit is None")

if __name__ == "__main__":
    asyncio.run(test_search_default_limit())

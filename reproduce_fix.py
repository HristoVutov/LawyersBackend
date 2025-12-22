
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append("c:/Users/vutov/Documents/PersonalProjects/Lawyers/LawyersBackend")

from app.middleware.virtual_fs import get_vfs
from app.tools.search_tools import search_indexed

async def test_search():
    print("Running search test...")
    
    # Initialize VFS
    vfs = get_vfs()
    vfs._project_path = Path("c:/Users/vutov/Documents/PersonalProjects/Lawyers/LawyersBackend/tests/fixtures/CaseProject")
    
    # Using a query that should match the test contract
    result = await search_indexed.ainvoke({"query": "test contract", "search_type": "keyword"})
    
    print("\n--- Search Result ---")
    print(result)
    
    print("\n--- Parsing Test ---")
    found_paths: list[str] = []
    if isinstance(result, str):
        for line in result.split("\n"):
            if line.strip().startswith("📄"):
                path = line.strip().replace("📄", "", 1).strip()
                found_paths.append(path)
                print(f"Parsed Path: {path}")
    
    if found_paths:
        print(f"\n✅ Success: Found {len(found_paths)} paths.")
    else:
        print("\n❌ Failure: No paths parsed.")

if __name__ == "__main__":
    asyncio.run(test_search())

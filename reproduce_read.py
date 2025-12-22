
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append("c:/Users/vutov/Documents/PersonalProjects/Lawyers/LawyersBackend")

from app.middleware.virtual_fs import get_vfs
from app.tools.file_tools import read_file

async def test_read_absolute_path():
    print("Running read_file test with absolute path...")
    
    # Initialize VFS
    vfs_root = Path("c:/Users/vutov/Documents/PersonalProjects/Lawyers/LawyersBackend/tests/fixtures/CaseProject")
    vfs = get_vfs()
    vfs.set_project(str(vfs_root))
    
    # Create a dummy file to read
    test_file = vfs_root / "test_read.txt"
    if not test_file.exists():
        test_file.write_text("Hello World", encoding="utf-8")
        
    abs_path = str(test_file.resolve())
    print(f"Reading absolute path: {abs_path}")
    
    # Try reading with absolute path
    result = await read_file.ainvoke({"file_path": abs_path})
    
    print("\n--- Read Result ---")
    print(result)
    
    if "Error" in result or "Invalid virtual path" in result:
        print("\n✅ Reproduced expected error.")
    else:
        print("\n❌ Failed to reproduce error (read succeeded).")

if __name__ == "__main__":
    asyncio.run(test_read_absolute_path())

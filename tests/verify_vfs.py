import asyncio
import os
import shutil
from pathlib import Path
from app.middleware.virtual_fs import set_project_path
from app.tools.file_tools import create_docx_file

async def verify_vfs_direct():
    print("🚀 Starting VFS Direct Verification...")
    
    # 1. Initialize VFS
    cwd = os.getcwd()
    set_project_path(cwd)
    print(f"📂 VFS Initialized with: {cwd}")
    
    # 2. Test create_docx_file with subdir
    thread_id = "test_thread_vfs"
    subdir = thread_id
    
    # Clean up
    output_dir = Path("output") / thread_id
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    content = "<h1>Test Document</h1><p>This is a test.</p>"
    filename = "vfs_test.docx"
    
    print(f"💾 Creating DOCX in subdir: {subdir}...")
    result = await create_docx_file.ainvoke({
        "file_name": filename, 
        "content": content, 
        "subdir": subdir
    })
    
    print(f"Result: {result}")
    
    # 3. Verify
    expected_path = output_dir / filename
    if expected_path.exists():
        print(f"✅ Success! File exists at {expected_path}")
    else:
        print(f"❌ Failed! File not found at {expected_path}")

if __name__ == "__main__":
    asyncio.run(verify_vfs_direct())

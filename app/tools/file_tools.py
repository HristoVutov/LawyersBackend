"""
File Tools - Stubbed for Server Deployment.
These tools are intended to be executed on the client-side via the Remote Bridge.
"""
import os
from pathlib import Path
from langchain_core.tools import tool

# --- Tools ---

@tool
async def read_file(file_path: str) -> str:
    """
    Read the contents of a file. 
    NOTE: This tool must be executed on the client-side via the Remote Bridge.
    """
    return f"Error: read_file must be executed on the client-side. Tool bridge not active for {file_path}."

@tool
async def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file. 
    NOTE: This tool must be executed on the client-side via the Remote Bridge.
    """
    return f"Error: write_file must be executed on the client-side. Tool bridge not active."

@tool
async def list_directory(dir_path: str = "/project/") -> str:
    """
    List contents of a virtual directory.
    NOTE: This tool must be executed on the client-side via the Remote Bridge.
    """
    return f"Error: list_directory must be executed on the client-side. Tool bridge not active."

@tool
async def glob(pattern: str, base_dir: str = "/project/") -> str:
    """
    Find files by pattern.
    NOTE: This tool must be executed on the client-side via the Remote Bridge.
    """
    return f"Error: glob must be executed on the client-side. Tool bridge not active."

@tool
async def grep(
    search_text: str,
    base_dir: str = "/project/",
    file_pattern: str = "*"
) -> str:
    """
    Search for text content within files.
    NOTE: This tool must be executed on the client-side via the Remote Bridge.
    """
    return f"Error: grep must be executed on the client-side. Tool bridge not active."

@tool
async def create_docx_file(file_name: str, content: str, subdir: str = "") -> str:
    """
    Create a new DOCX file.
    NOTE: This tool must be executed on the client-side via the Remote Bridge.
    """
    return f"Error: create_docx_file must be executed on the client-side. Tool bridge not active."

# Export all tools
FILE_TOOLS = [
    read_file,
    write_file,
    list_directory,
    glob,
    grep,
    create_docx_file,
]

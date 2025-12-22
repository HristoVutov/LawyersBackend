"""
File Tools - File system operations for agents using virtual paths.

Provides tools for:
- Reading and writing files (via virtual paths)
- Listing directory contents
- Globbing for file patterns
- Grep for searching file contents
- Creating DOCX files from HTML

Virtual Paths:
- /templates/ -> User's Documents/LawyersAI/Templates (read/write)
- /project/   -> Current project folder (read only)
- /output/    -> Project/.generatedFiles (read/write)
"""
import os
from pathlib import Path
from langchain_core.tools import tool

from app.middleware.virtual_fs import (
    get_vfs, resolve_path, can_write, to_virtual
)


# --- Helper Functions ---

async def glob_files(base_dir: Path, pattern: str, vfs_base: str = "") -> list[str]:
    """
    Recursive glob-like file search.
    
    Args:
        base_dir: Directory to start search from (real path)
        pattern: File pattern like *.json, *.pdf, *search*
        vfs_base: Virtual base path for results (e.g., "/project/")
    
    Returns:
        List of virtual paths matching the pattern
    """
    results = []
    base_path = Path(base_dir)
    
    # Extract extension from pattern
    if pattern.startswith("*"):
        ext = pattern[1:]  # e.g., ".json" from "*.json"
    else:
        ext = pattern
    
    def should_skip(name: str) -> bool:
        return name.startswith(".") or name == "node_modules" or name == "__pycache__"
    
    def walk(dir_path: Path, virtual_prefix: str):
        try:
            for item in dir_path.iterdir():
                if item.is_dir() and not should_skip(item.name):
                    walk(item, f"{virtual_prefix}{item.name}/")
                elif item.is_file():
                    if pattern == "*" or item.name.endswith(ext) or ext.replace("*", "") in item.name:
                        results.append(f"{virtual_prefix}{item.name}")
        except (PermissionError, OSError):
            pass  # Skip inaccessible dirs
    
    walk(base_path, vfs_base)
    return results


def get_output_path() -> Path:
    """Get /output/ folder real path."""
    return resolve_path("/output/")


async def ensure_output_folder() -> Path | None:
    """Ensure /output/ folder exists."""
    try:
        output_path = get_output_path()
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    except Exception as e:
        print(f"Failed to access output folder: {e}")
        return None


# --- Tools ---

@tool
async def read_file(file_path: str) -> str:
    """
    Read the contents of a file using virtual paths.
    
    Virtual paths:
    - /project/filename.pdf - Read from current project
    - /templates/contract.txt - Read from templates
    - /output/draft.docx - Read from generated files
    
    Args:
        file_path: Virtual path to the file (e.g., /project/document.pdf)
    """
    try:
        # Support absolute paths by converting to virtual if possible
        if os.path.isabs(file_path) or ":" in file_path:
            virtual = to_virtual(file_path)
            if virtual:
                file_path = virtual

        real_path = resolve_path(file_path)
        
        if not real_path.exists():
            print(f"[TOOL ERROR] File not found: {file_path}")
            return f"Error: File not found: {file_path}"
        
        if real_path.suffix.lower() in ['.pdf', '.docx', '.doc', '.xlsx', '.xls']:
            # CHECK FOR INDEXED CONTENT FIRST
            try:
                # 1. Check in .indexedfiles directory (standard location)
                from app.config import get_settings
                settings = get_settings()
                
                # Check .indexedfiles/filename.index.json
                index_path = settings.indexed_files_dir / f"{real_path.name}.index.json"
                if not index_path.exists():
                    # 2. Check in same directory
                    index_path = real_path.parent / f"{real_path.name}.index.json"
                
                if index_path.exists():
                    import json
                    data = json.loads(index_path.read_text(encoding="utf-8"))
                    content = data.get("content")
                    if content:
                        return f"[Content from Index for {real_path.name}]\n{content}"
            except Exception as e:
                print(f"[TOOL WARNING] Failed to read index for {file_path}: {e}")

            return f"Binary file at {file_path}. Use appropriate tools to process."
        
        content = real_path.read_text(encoding="utf-8")
        return content
    except ValueError as e:
        print(f"[TOOL ERROR] Invalid path: {str(e)}")
        return f"Error: {str(e)}"
    except Exception as e:
        print(f"[TOOL ERROR] read_file failed: {str(e)}")
        return f"Error reading file: {str(e)}"


@tool
async def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file. Only /output/ and /templates/ are writable.
    
    Args:
        file_path: Virtual path to write to (e.g., /output/notes.txt)
        content: The content to write to the file
    """
    try:
        # Check permissions
        if not can_write(file_path):
            return f"Error: Cannot write to {file_path}. Only /output/ and /templates/ are writable."
        
        real_path = resolve_path(file_path)
        real_path.parent.mkdir(parents=True, exist_ok=True)
        real_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {file_path}"
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool
async def list_directory(dir_path: str = "/project/") -> str:
    """
    List contents of a virtual directory.
    
    Args:
        dir_path: Virtual directory path (e.g., /project/, /templates/, /output/)
    """
    try:
        real_path = resolve_path(dir_path)
        
        if not real_path.exists():
            return f"Directory not found: {dir_path}"
        
        items = []
        for item in real_path.iterdir():
            if item.name.startswith("."):
                continue
            prefix = "📁" if item.is_dir() else "📄"
            # Return full virtual path to avoid agent confusion
            # Ensure dir_path ends with /
            base = dir_path if dir_path.endswith("/") else f"{dir_path}/"
            items.append(f"{prefix} {base}{item.name}")
        
        if not items:
            return f"Empty directory: {dir_path}"
        
        return f"Contents of {dir_path}:\n" + "\n".join(sorted(items))
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error listing directory: {str(e)}"


@tool
async def glob(pattern: str, base_dir: str = "/project/") -> str:
    """
    Find files by pattern within a virtual directory.
    
    Args:
        pattern: File pattern like *.json, *.docx, *.pdf
        base_dir: Virtual base directory (e.g., /project/, /templates/)
    """
    try:
        real_path = resolve_path(base_dir)
        # Normalize base_dir for virtual results
        vfs_base = base_dir if base_dir.endswith("/") else base_dir + "/"
        
        files = await glob_files(real_path, pattern, vfs_base)
        
        if not files:
            return f"No files found matching pattern: {pattern} in {base_dir}"
        
        result = "\n".join(files[:50])
        if len(files) > 50:
            result += f"\n... and {len(files) - 50} more"
        return result
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error searching files: {str(e)}"


@tool
async def grep(
    search_text: str,
    base_dir: str = "/project/",
    file_pattern: str = "*"
) -> str:
    """
    Search for text content within files. Returns virtual file paths and line numbers.
    
    Args:
        search_text: Text to search for
        base_dir: Virtual base directory to search (e.g., /project/)
        file_pattern: File pattern like *.json, *.txt
    """
    try:
        real_base = resolve_path(base_dir)
        vfs_base = base_dir if base_dir.endswith("/") else base_dir + "/"
        
        files = await glob_files(real_base, file_pattern, vfs_base)
        
        results = []
        search_lower = search_text.lower()
        
        for virtual_path in files[:100]:
            try:
                real_path = resolve_path(virtual_path)
                content = real_path.read_text(encoding="utf-8", errors="ignore")
                lines = content.split("\n")
                
                for i, line in enumerate(lines):
                    if search_lower in line.lower():
                        results.append({
                            "file": virtual_path,
                            "line": i + 1,
                            "content": line[:200]
                        })
                        if len(results) >= 20:
                            break
            except Exception:
                pass  # Skip unreadable files
            
            if len(results) >= 20:
                break
        
        if not results:
            return f"No matches found for: {search_text}"
        
        return "\n".join(f"{r['file']}:{r['line']}: {r['content']}" for r in results)
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error searching: {str(e)}"


@tool
async def create_docx_file(file_name: str, content: str) -> str:
    """
    Create a new DOCX file in /output/ folder.
    Use this for generating legal documents, contracts, letters, etc.
    
    Args:
        file_name: Name for the file (e.g., 'contract', 'letter'). Will be saved as .docx
        content: The document content in HTML format
    """
    try:
        from docx import Document
        import re
        
        output_path = await ensure_output_folder()
        if not output_path:
            return "Error: Could not access /output/ folder. Is a project selected?"
        
        # Clean up file name
        base_name = re.sub(r"\.(docx|html|txt|md)$", "", file_name, flags=re.IGNORECASE)
        docx_path = output_path / f"{base_name}.docx"
        html_path = output_path / f"{base_name}.docx.html"
        
        # Wrap content in HTML structure
        full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Calibri, sans-serif; font-size: 11pt; }}
        h1 {{ font-size: 16pt; font-weight: bold; }}
        h2 {{ font-size: 14pt; font-weight: bold; }}
        h3 {{ font-size: 12pt; font-weight: bold; }}
    </style>
</head>
<body>{content}</body>
</html>"""
        
        # Save HTML file for reference/editing
        html_path.write_text(full_html, encoding="utf-8")
        
        # Create DOCX using python-docx
        doc = Document()
        
        # Simple HTML to DOCX conversion
        text_content = re.sub(r"<[^>]+>", "\n", content)
        text_content = re.sub(r"\n+", "\n", text_content).strip()
        
        for line in text_content.split("\n"):
            if line.strip():
                doc.add_paragraph(line.strip())
        
        doc.save(str(docx_path))
        
        return f"""✅ DOCX file created successfully!
📄 DOCX: /output/{base_name}.docx
📄 HTML: /output/{base_name}.docx.html"""
    except ImportError:
        return "Error: python-docx not installed. Run: pip install python-docx"
    except Exception as e:
        return f"Error creating DOCX file: {str(e)}"


# Export all tools
FILE_TOOLS = [
    read_file,
    write_file,
    list_directory,
    glob,
    grep,
    create_docx_file,
]

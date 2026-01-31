"""
Legal Tools - Document reading and template operations for legal documents.

Provides tools for:
- Reading documents (.docx, .pdf, .txt)
- Filling templates with placeholders
"""
import re
from pathlib import Path
from langchain_core.tools import tool

from app.config import get_settings
from app.middleware.virtual_fs import get_vfs, resolve_path
from app.services import (
    session_store, 
    mark_document_as_read, 
    is_document_read,
    cache_document_content,
    get_cached_document_content
)


async def get_index_file_content(file_path: Path) -> str | None:
    """
    Check if an index file exists for the document (contains pre-extracted text).
    
    Looks in:
    1. Same directory: file.pdf.index.json
    2. indexedfiles folder: indexedfiles/file.pdf.index.json
    """
    settings = get_settings()
    file_name = file_path.name
    
    potential_paths = [
        file_path.parent / f"{file_name}.index.json",
        settings.indexed_files_dir / f"{file_name}.index.json",
        file_path.parent / "indexedfiles" / f"{file_name}.index.json",
    ]
    
    for index_path in potential_paths:
        try:
            if index_path.exists():
                import json
                content = index_path.read_text(encoding="utf-8")
                data = json.loads(content)
                if data.get("content"):
                    return f"[Read from Index]:\n{data['content']}"
        except Exception:
            continue
    
    return None


@tool
async def read_document(file_path: str, conversation_id: str = "default") -> str:
    """
    Read text from document files (.docx, .pdf, .txt, .md).
    NOTE: This tool must be executed on the client-side via the Remote Bridge.
    """
    return f"Error: read_document must be executed on the client-side. Tool bridge not active for {file_path}."


@tool
async def fill_template(template_content: str, replacements: str) -> str:
    """
    Replace placeholders like {{NAME}} in a template string with provided values.
    
    Args:
        template_content: The original template content (HTML/Text)
        replacements: JSON string of replacements, e.g. '{"CLIENT": "John Doe"}'
    """
    try:
        import json
        
        result = template_content
        
        # Handle JSON string input
        if isinstance(replacements, str):
            data = json.loads(replacements)
        else:
            data = replacements
        
        for key, value in data.items():
            # Replace {{KEY}} pattern
            pattern = r"\{\{" + re.escape(key) + r"\}\}"
            result = re.sub(pattern, str(value), result)
        
        return result
        
    except json.JSONDecodeError:
        return "Error: Replacements must be a valid JSON object."
    except Exception as e:
        return f"Error filling template: {str(e)}"


# Export all tools
LEGAL_TOOLS = [
    read_document,
    fill_template,
]

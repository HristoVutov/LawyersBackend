"""
Filesystem Middleware - Deep Agents Pattern.

Implements the LangChain Deep Agents FilesystemMiddleware pattern:
- Provides a virtual filesystem for agents to store/retrieve data
- Supports short-term (state) and long-term (store) backends
- Tools: ls, read_file, write_file, edit_file

Reference: https://docs.langchain.com/oss/python/deepagents/middleware#filesystem-middleware
"""
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any
from datetime import datetime
from langchain_core.tools import tool
from app.middleware.prompts.filesystem_prompt import FILESYSTEM_SYSTEM_PROMPT


@dataclass
class VirtualFile:
    """A virtual file in the filesystem."""
    content: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    size: int = 0
    
    def __post_init__(self):
        self.size = len(self.content)


class StateBackend:
    """
    Short-term state backend - stores files in memory per conversation.
    Data is lost when the conversation ends.
    """
    
    def __init__(self):
        self._stores: dict[str, dict[str, VirtualFile]] = {}
    
    def _get_store(self, conversation_id: str) -> dict[str, VirtualFile]:
        if conversation_id not in self._stores:
            self._stores[conversation_id] = {}
        return self._stores[conversation_id]
    
    def list_files(self, conversation_id: str, path: str = "/") -> list[str]:
        """List files at a given path."""
        store = self._get_store(conversation_id)
        prefix = path if path.endswith("/") else path + "/"
        
        files = []
        dirs = set()
        
        for file_path in store.keys():
            if file_path.startswith(prefix):
                relative = file_path[len(prefix):]
                if "/" in relative:
                    # It's in a subdirectory
                    dir_name = relative.split("/")[0]
                    dirs.add(dir_name + "/")
                else:
                    # Direct child
                    files.append(relative)
        
        return sorted(list(dirs)) + sorted(files)
    
    def read_file(self, conversation_id: str, path: str) -> str | None:
        """Read a file's content."""
        store = self._get_store(conversation_id)
        vfile = store.get(path)
        return vfile.content if vfile else None
    
    def write_file(self, conversation_id: str, path: str, content: str) -> bool:
        """Write content to a file."""
        store = self._get_store(conversation_id)
        store[path] = VirtualFile(content=content)
        return True
    
    def delete_file(self, conversation_id: str, path: str) -> bool:
        """Delete a file."""
        store = self._get_store(conversation_id)
        if path in store:
            del store[path]
            return True
        return False
    
    def exists(self, conversation_id: str, path: str) -> bool:
        """Check if a file exists."""
        store = self._get_store(conversation_id)
        return path in store


# Global state backend
_state_backend = StateBackend()

# Current conversation ID
_current_conversation_id = "default"


def set_filesystem_conversation_id(conversation_id: str) -> None:
    """Set the current conversation ID for filesystem operations."""
    global _current_conversation_id
    _current_conversation_id = conversation_id


@tool
def fs_ls(path: str = "/") -> str:
    """
    List files in a directory of the virtual filesystem.
    
    Args:
        path: Directory path to list (default: root "/")
    """
    try:
        files = _state_backend.list_files(_current_conversation_id, path)
        if not files:
            return f"Directory '{path}' is empty or does not exist."
        return "\n".join(files)
    except Exception as e:
        return f"Error listing directory: {str(e)}"


@tool
def fs_read_file(path: str, start_line: int | None = None, num_lines: int | None = None) -> str:
    """
    Read file content from the virtual filesystem.
    
    Args:
        path: Path to the file to read
        start_line: Optional line to start reading from (1-indexed)
        num_lines: Optional number of lines to read
    """
    try:
        content = _state_backend.read_file(_current_conversation_id, path)
        if content is None:
            return f"File not found: {path}"
        
        if start_line is not None:
            lines = content.split("\n")
            start_idx = max(0, start_line - 1)
            end_idx = len(lines) if num_lines is None else start_idx + num_lines
            content = "\n".join(lines[start_idx:end_idx])
        
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def fs_write_file(path: str, content: str) -> str:
    """
    Write content to a file in the virtual filesystem.
    Creates the file if it doesn't exist, overwrites if it does.
    
    Args:
        path: Path to the file to write
        content: Content to write to the file
    """
    try:
        _state_backend.write_file(_current_conversation_id, path, content)
        return f"Successfully wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool
def fs_edit_file(path: str, old_content: str, new_content: str) -> str:
    """
    Edit a file by replacing specific content.
    
    Args:
        path: Path to the file to edit
        old_content: The exact content to replace
        new_content: The new content to insert
    """
    try:
        content = _state_backend.read_file(_current_conversation_id, path)
        if content is None:
            return f"File not found: {path}"
        
        if old_content not in content:
            return f"Content to replace not found in file."
        
        new_file_content = content.replace(old_content, new_content, 1)
        _state_backend.write_file(_current_conversation_id, path, new_file_content)
        
        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error editing file: {str(e)}"


class FilesystemMiddleware:
    """
    Deep Agents FilesystemMiddleware pattern.
    
    Provides a virtual filesystem for agents to store working data.
    
    Usage:
        middleware = FilesystemMiddleware()
        
        # Before agent call, set conversation:
        middleware.before(message, conversation_id)
        
        # Agent gets middleware.tools
    """
    
    DEFAULT_SYSTEM_PROMPT = FILESYSTEM_SYSTEM_PROMPT
    
    def __init__(
        self,
        system_prompt: str | None = None,
        custom_tool_descriptions: dict[str, str] | None = None
    ):
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.custom_descriptions = custom_tool_descriptions or {}
        self.tools = [fs_ls, fs_read_file, fs_write_file, fs_edit_file]
    
    def before(self, message: str, conversation_id: str) -> str:
        """
        Called before each agent invocation.
        Sets the conversation ID for filesystem operations.
        """
        set_filesystem_conversation_id(conversation_id)
        return message
    
    def get_system_prompt_addition(self) -> str:
        """Get the system prompt addition for filesystem."""
        return self.system_prompt
    
    def get_filesystem_context(self, conversation_id: str) -> str:
        """Get current filesystem state for context injection."""
        files = _state_backend.list_files(conversation_id, "/")
        if not files:
            return ""
        
        return f"[VIRTUAL FILESYSTEM]\nFiles: {', '.join(files)}\n\n"

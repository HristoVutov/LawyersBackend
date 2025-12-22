"""
Deep Agents Backends - Storage backends for the virtual filesystem.

Implements the LangChain Deep Agents backend pattern:
- StateBackend: Ephemeral per-conversation storage
- StoreBackend: Cross-thread persistent storage (LangGraph Store)
- FilesystemBackend: Real disk access with sandboxing
- CompositeBackend: Routes paths to different backends

Reference: https://docs.langchain.com/oss/python/deepagents/backends
"""
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from app.config import get_settings


@dataclass
class VirtualFile:
    """A virtual file in the filesystem."""
    content: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def size(self) -> int:
        return len(self.content)
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "size": self.size,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "VirtualFile":
        return cls(
            content=data.get("content", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            modified_at=data.get("modified_at", datetime.now().isoformat()),
        )


class BackendProtocol(ABC):
    """Base protocol for all backends."""
    
    @abstractmethod
    def list_files(self, path: str = "/") -> list[str]:
        """List files at a given path."""
        pass
    
    @abstractmethod
    def read_file(self, path: str) -> str | None:
        """Read a file's content."""
        pass
    
    @abstractmethod
    def write_file(self, path: str, content: str) -> bool:
        """Write content to a file."""
        pass
    
    @abstractmethod
    def delete_file(self, path: str) -> bool:
        """Delete a file."""
        pass
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a file exists."""
        pass


class StateBackend(BackendProtocol):
    """
    StateBackend - Ephemeral per-conversation storage.
    
    Stores files in memory, lost when the conversation ends.
    Used for scratch notes and intermediate results.
    """
    
    def __init__(self, conversation_id: str = "default"):
        self.conversation_id = conversation_id
        self._files: dict[str, VirtualFile] = {}
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path to standard format."""
        if not path.startswith("/"):
            path = "/" + path
        return str(PurePosixPath(path))
    
    def list_files(self, path: str = "/") -> list[str]:
        path = self._normalize_path(path)
        prefix = path if path.endswith("/") else path + "/"
        
        files = []
        dirs = set()
        
        for file_path in self._files.keys():
            if file_path.startswith(prefix):
                relative = file_path[len(prefix):]
                if "/" in relative:
                    dir_name = relative.split("/")[0]
                    dirs.add(dir_name + "/")
                else:
                    files.append(relative)
        
        return sorted(list(dirs)) + sorted(files)
    
    def read_file(self, path: str) -> str | None:
        path = self._normalize_path(path)
        vfile = self._files.get(path)
        return vfile.content if vfile else None
    
    def write_file(self, path: str, content: str) -> bool:
        path = self._normalize_path(path)
        self._files[path] = VirtualFile(content=content)
        return True
    
    def delete_file(self, path: str) -> bool:
        path = self._normalize_path(path)
        if path in self._files:
            del self._files[path]
            return True
        return False
    
    def exists(self, path: str) -> bool:
        path = self._normalize_path(path)
        return path in self._files


class StoreBackend(BackendProtocol):
    """
    StoreBackend - Cross-thread persistent storage.
    
    Uses a JSON file to persist data across conversations.
    Suitable for long-term memories, entities, learned preferences.
    """
    
    def __init__(self, store_path: Path | str | None = None):
        settings = get_settings()
        self.store_path = Path(store_path) if store_path else settings.project_root / ".agent_store.json"
        self._files: dict[str, VirtualFile] = {}
        self._load()
    
    def _load(self) -> None:
        """Load store from disk."""
        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text(encoding="utf-8"))
                for path, file_data in data.get("files", {}).items():
                    self._files[path] = VirtualFile.from_dict(file_data)
            except Exception as e:
                print(f"⚠️ StoreBackend load error: {e}")
    
    def _save(self) -> None:
        """Persist store to disk."""
        try:
            data = {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "files": {path: vf.to_dict() for path, vf in self._files.items()}
            }
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            self.store_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"⚠️ StoreBackend save error: {e}")
    
    def _normalize_path(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return str(PurePosixPath(path))
    
    def list_files(self, path: str = "/") -> list[str]:
        path = self._normalize_path(path)
        prefix = path if path.endswith("/") else path + "/"
        
        files = []
        dirs = set()
        
        for file_path in self._files.keys():
            if file_path.startswith(prefix):
                relative = file_path[len(prefix):]
                if "/" in relative:
                    dir_name = relative.split("/")[0]
                    dirs.add(dir_name + "/")
                else:
                    files.append(relative)
        
        return sorted(list(dirs)) + sorted(files)
    
    def read_file(self, path: str) -> str | None:
        path = self._normalize_path(path)
        vfile = self._files.get(path)
        return vfile.content if vfile else None
    
    def write_file(self, path: str, content: str) -> bool:
        path = self._normalize_path(path)
        self._files[path] = VirtualFile(content=content)
        self._save()
        return True
    
    def delete_file(self, path: str) -> bool:
        path = self._normalize_path(path)
        if path in self._files:
            del self._files[path]
            self._save()
            return True
        return False
    
    def exists(self, path: str) -> bool:
        path = self._normalize_path(path)
        return path in self._files


class FilesystemBackend(BackendProtocol):
    """
    FilesystemBackend - Real disk access with sandboxing.
    
    Provides access to actual files on disk, sandboxed to root_dir.
    """
    
    def __init__(self, root_dir: Path | str | None = None, virtual_mode: bool = True):
        settings = get_settings()
        self.root_dir = Path(root_dir) if root_dir else settings.project_root
        self.virtual_mode = virtual_mode  # Sandbox paths under root_dir
    
    def _resolve_path(self, path: str) -> Path:
        """Safely resolve path within root_dir."""
        # Remove leading slash for joining
        clean_path = path.lstrip("/")
        resolved = (self.root_dir / clean_path).resolve()
        
        # Security: ensure we're still under root_dir
        if self.virtual_mode:
            try:
                resolved.relative_to(self.root_dir.resolve())
            except ValueError:
                raise ValueError(f"Path escapes sandbox: {path}")
        
        return resolved
    
    def list_files(self, path: str = "/") -> list[str]:
        try:
            resolved = self._resolve_path(path)
            if not resolved.exists() or not resolved.is_dir():
                return []
            
            items = []
            for item in resolved.iterdir():
                if item.name.startswith("."):
                    continue
                if item.is_dir():
                    items.append(item.name + "/")
                else:
                    items.append(item.name)
            
            return sorted(items)
        except Exception:
            return []
    
    def read_file(self, path: str) -> str | None:
        try:
            resolved = self._resolve_path(path)
            if resolved.exists() and resolved.is_file():
                return resolved.read_text(encoding="utf-8", errors="ignore")
            return None
        except Exception:
            return None
    
    def write_file(self, path: str, content: str) -> bool:
        try:
            resolved = self._resolve_path(path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            return True
        except Exception:
            return False
    
    def delete_file(self, path: str) -> bool:
        try:
            resolved = self._resolve_path(path)
            if resolved.exists():
                resolved.unlink()
                return True
            return False
        except Exception:
            return False
    
    def exists(self, path: str) -> bool:
        try:
            resolved = self._resolve_path(path)
            return resolved.exists()
        except Exception:
            return False


class CompositeBackend(BackendProtocol):
    """
    CompositeBackend - Routes paths to different backends.
    
    Allows combining multiple backends based on path prefix.
    
    Example:
        CompositeBackend(
            default=StateBackend(),
            routes={
                "/memories/": StoreBackend(),
                "/documents/": FilesystemBackend(root_dir="indexedfiles"),
            }
        )
    """
    
    def __init__(
        self,
        default: BackendProtocol,
        routes: dict[str, BackendProtocol] | None = None
    ):
        self.default = default
        self.routes = routes or {}
    
    def _get_backend(self, path: str) -> tuple[BackendProtocol, str]:
        """Get the appropriate backend and adjusted path."""
        for prefix, backend in sorted(self.routes.items(), key=lambda x: -len(x[0])):
            if path.startswith(prefix):
                # Keep the path as-is for the routed backend
                return backend, path
        return self.default, path
    
    def list_files(self, path: str = "/") -> list[str]:
        backend, adjusted_path = self._get_backend(path)
        return backend.list_files(adjusted_path)
    
    def read_file(self, path: str) -> str | None:
        backend, adjusted_path = self._get_backend(path)
        return backend.read_file(adjusted_path)
    
    def write_file(self, path: str, content: str) -> bool:
        backend, adjusted_path = self._get_backend(path)
        return backend.write_file(adjusted_path, content)
    
    def delete_file(self, path: str) -> bool:
        backend, adjusted_path = self._get_backend(path)
        return backend.delete_file(adjusted_path)
    
    def exists(self, path: str) -> bool:
        backend, adjusted_path = self._get_backend(path)
        return backend.exists(adjusted_path)
    
    def list_all_routes(self) -> list[str]:
        """List all configured route prefixes."""
        return list(self.routes.keys())


def create_default_backend(conversation_id: str = "default") -> CompositeBackend:
    """
    Create the default composite backend for the legal assistant.
    
    Routes:
        /               → StateBackend (ephemeral scratch)
        /memories/      → StoreBackend (persistent entities, history)
        /documents/     → FilesystemBackend (real indexedfiles)
    """
    settings = get_settings()
    
    return CompositeBackend(
        default=StateBackend(conversation_id),
        routes={
            "/memories/": StoreBackend(),
            "/documents/": FilesystemBackend(root_dir=settings.indexed_files_dir),
        }
    )

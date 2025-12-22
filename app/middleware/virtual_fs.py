"""
Virtual File System - Path mapping and permissions for agent file access.

Provides a virtual directory abstraction that:
- Maps virtual paths (/templates/, /project/, /output/) to real filesystem paths
- Enforces read/write permissions per mount point
- Prevents path traversal attacks
"""
import os
from pathlib import Path
from enum import Enum
from typing import Optional


class Permission(Enum):
    """File access permissions."""
    READ = "read"
    READ_WRITE = "read_write"


class VirtualFileSystem:
    """
    Virtual file system with mount points and permission enforcement.
    
    Virtual paths:
    - /templates/ -> User's Documents/LawyersAI/Templates (read/write)
    - /project/   -> Selected project folder (read only)
    - /output/    -> Project/.generatedFiles (read/write)
    """
    
    def __init__(self, project_path: Optional[str] = None):
        self._mounts: dict[str, tuple[Path, Permission]] = {}
        self._project_path: Optional[Path] = None
        
        # Always mount templates (global location)
        templates_path = Path.home() / "Documents" / "LawyersAI" / "Templates"
        templates_path.mkdir(parents=True, exist_ok=True)
        self.mount("/templates/", templates_path, Permission.READ_WRITE)
        
        # Mount project if provided
        if project_path:
            self.set_project(project_path)
    
    def mount(self, virtual_path: str, real_path: Path, permission: Permission) -> None:
        """
        Mount a real path to a virtual path.
        
        Args:
            virtual_path: Virtual path like /templates/
            real_path: Real filesystem path
            permission: READ or READ_WRITE
        """
        # Normalize virtual path
        virtual_path = self._normalize_virtual(virtual_path)
        self._mounts[virtual_path] = (real_path, permission)
    
    def set_project(self, project_path: str) -> None:
        """
        Set the active project path and update mounts.
        
        Args:
            project_path: Absolute path to project folder
        """
        self._project_path = Path(project_path)
        
        # Mount project root as read-only
        self.mount("/project/", self._project_path, Permission.READ)
        
        # Mount .generatedFiles as read-write
        output_path = self._project_path / ".generatedFiles"
        output_path.mkdir(parents=True, exist_ok=True)
        self.mount("/output/", output_path, Permission.READ_WRITE)
        
        # Mount .indexedfiles as read-only (agents can inspect, but only Indexer writes)
        index_path = self._project_path / ".indexedfiles"
        index_path.mkdir(parents=True, exist_ok=True)
        self.mount("/indexes/", index_path, Permission.READ)
    
    def resolve(self, virtual_path: str) -> Path:
        """
        Convert a virtual path to a real filesystem path.
        
        Args:
            virtual_path: Path like /project/document.pdf
            
        Returns:
            Resolved real filesystem Path
            
        Raises:
            ValueError: If path is invalid or outside mount points
        """
        virtual_path = self._normalize_virtual(virtual_path)
        
        # Find matching mount
        mount_point, real_base, _ = self._find_mount(virtual_path)
        
        # Get relative path within mount
        relative = virtual_path[len(mount_point):]
        
        # Resolve and validate no escape
        resolved = (real_base / relative).resolve()
        
        # Security: ensure resolved path is within mount
        try:
            resolved.relative_to(real_base.resolve())
        except ValueError:
            raise ValueError(f"Path '{virtual_path}' escapes mount point")
        
        return resolved
    
    def can_write(self, virtual_path: str) -> bool:
        """
        Check if writing is allowed to this virtual path.
        
        Args:
            virtual_path: Path to check
            
        Returns:
            True if writing is allowed
        """
        try:
            virtual_path = self._normalize_virtual(virtual_path)
            _, _, permission = self._find_mount(virtual_path)
            return permission == Permission.READ_WRITE
        except ValueError:
            return False
    
    def list_mounts(self) -> dict[str, dict]:
        """
        List all mount points and their info.
        
        Returns:
            Dict mapping virtual paths to {real_path, permission, exists}
        """
        result = {}
        for vpath, (rpath, perm) in self._mounts.items():
            result[vpath] = {
                "real_path": str(rpath),
                "permission": perm.value,
                "exists": rpath.exists(),
            }
        return result
    
    def to_virtual(self, real_path: str | Path) -> Optional[str]:
        """
        Convert a real path back to a virtual path if possible.
        
        Args:
            real_path: Real filesystem path
            
        Returns:
            Virtual path or None if not in any mount
        """
        real_path = Path(real_path).resolve()
        
        for vpath, (rpath, _) in self._mounts.items():
            try:
                relative = real_path.relative_to(rpath.resolve())
                return vpath + str(relative).replace("\\", "/")
            except ValueError:
                continue
        
        return None
    
    def _normalize_virtual(self, path: str) -> str:
        """Normalize a virtual path."""
        # Ensure starts with /
        if not path.startswith("/"):
            path = "/" + path
        # Convert backslashes
        path = path.replace("\\", "/")
        # Remove double slashes
        while "//" in path:
            path = path.replace("//", "/")
        return path
    
    def _find_mount(self, virtual_path: str) -> tuple[str, Path, Permission]:
        """
        Find the mount point for a virtual path.
        
        Returns:
            Tuple of (mount_point, real_base_path, permission)
            
        Raises:
            ValueError: If no matching mount found
        """
        for mount_point, (real_path, permission) in self._mounts.items():
            if virtual_path.startswith(mount_point) or virtual_path == mount_point.rstrip("/"):
                return mount_point, real_path, permission
        
        raise ValueError(
            f"Invalid virtual path: '{virtual_path}'. "
            f"Must start with one of: {list(self._mounts.keys())}"
        )


# Global instance - will be configured when project is set
_vfs: Optional[VirtualFileSystem] = None


def get_vfs() -> VirtualFileSystem:
    """Get the global VirtualFileSystem instance."""
    global _vfs
    if _vfs is None:
        _vfs = VirtualFileSystem()
    return _vfs


def set_project_path(project_path: str) -> None:
    """Set the active project path globally."""
    get_vfs().set_project(project_path)


def resolve_path(virtual_path: str) -> Path:
    """Resolve a virtual path to real path."""
    return get_vfs().resolve(virtual_path)


def can_write(virtual_path: str) -> bool:
    """Check if writing is allowed."""
    return get_vfs().can_write(virtual_path)


def to_virtual(real_path: str | Path) -> Optional[str]:
    """Convert real path to virtual."""
    return get_vfs().to_virtual(real_path)

"""
Tests for Virtual File System.

Tests path resolution, permission enforcement, and path traversal prevention.
"""
import pytest
from pathlib import Path
import tempfile
import shutil

from app.middleware.virtual_fs import (
    VirtualFileSystem, Permission, 
    get_vfs, set_project_path, resolve_path, can_write, to_virtual
)


class TestVirtualFileSystem:
    """Test VirtualFileSystem class."""
    
    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project structure."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        
        # Create some files
        (project_dir / "document.pdf").write_text("test content")
        (project_dir / "subfolder").mkdir()
        (project_dir / "subfolder" / "nested.txt").write_text("nested content")
        
        return project_dir
    
    @pytest.fixture
    def vfs(self, temp_project):
        """Create VFS with temp project."""
        return VirtualFileSystem(str(temp_project))
    
    def test_templates_mounted_by_default(self):
        """Templates should always be mounted."""
        vfs = VirtualFileSystem()
        mounts = vfs.list_mounts()
        assert "/templates/" in mounts
        assert mounts["/templates/"]["permission"] == "read_write"
    
    def test_project_mount(self, vfs, temp_project):
        """Project should be mounted as read-only."""
        mounts = vfs.list_mounts()
        assert "/project/" in mounts
        assert mounts["/project/"]["real_path"] == str(temp_project)
        assert mounts["/project/"]["permission"] == "read"
    
    def test_output_mount(self, vfs, temp_project):
        """Output should be mounted as read-write."""
        mounts = vfs.list_mounts()
        assert "/output/" in mounts
        assert mounts["/output/"]["permission"] == "read_write"
    
    def test_resolve_project_path(self, vfs, temp_project):
        """Resolve virtual path to real path."""
        resolved = vfs.resolve("/project/document.pdf")
        assert resolved == temp_project / "document.pdf"
    
    def test_resolve_nested_path(self, vfs, temp_project):
        """Resolve nested virtual path."""
        resolved = vfs.resolve("/project/subfolder/nested.txt")
        assert resolved == temp_project / "subfolder" / "nested.txt"
    
    def test_resolve_invalid_mount(self, vfs):
        """Invalid mount should raise error."""
        with pytest.raises(ValueError, match="Invalid virtual path"):
            vfs.resolve("/invalid/path.txt")
    
    def test_path_traversal_blocked(self, vfs):
        """Path traversal attacks should be blocked."""
        with pytest.raises(ValueError, match="escapes mount point"):
            vfs.resolve("/project/../../../etc/passwd")
    
    def test_can_write_project(self, vfs):
        """Writing to project should be blocked."""
        assert vfs.can_write("/project/file.txt") is False
    
    def test_can_write_output(self, vfs):
        """Writing to output should be allowed."""
        assert vfs.can_write("/output/file.txt") is True
    
    def test_can_write_templates(self, vfs):
        """Writing to templates should be allowed."""
        assert vfs.can_write("/templates/template.docx") is True
    
    def test_to_virtual_conversion(self, vfs, temp_project):
        """Real path should convert back to virtual."""
        real_path = temp_project / "document.pdf"
        virtual = vfs.to_virtual(real_path)
        assert virtual == "/project/document.pdf"
    
    def test_set_project_updates_mounts(self, tmp_path):
        """Setting project should update mounts."""
        vfs = VirtualFileSystem()
        
        # Initially no project mount
        project1 = tmp_path / "project1"
        project1.mkdir()
        
        vfs.set_project(str(project1))
        assert vfs.resolve("/project/") == project1
        
        # Change project
        project2 = tmp_path / "project2"
        project2.mkdir()
        
        vfs.set_project(str(project2))
        assert vfs.resolve("/project/") == project2


class TestGlobalFunctions:
    """Test module-level convenience functions."""
    
    def test_get_vfs_singleton(self):
        """get_vfs should return same instance."""
        vfs1 = get_vfs()
        vfs2 = get_vfs()
        assert vfs1 is vfs2
    
    def test_set_project_path(self, tmp_path):
        """set_project_path should update global VFS."""
        project = tmp_path / "test"
        project.mkdir()
        
        set_project_path(str(project))
        
        vfs = get_vfs()
        assert vfs._project_path == project
    
    def test_can_write_function(self, tmp_path):
        """can_write function should work correctly."""
        project = tmp_path / "test"
        project.mkdir()
        set_project_path(str(project))
        
        assert can_write("/project/file.txt") is False
        assert can_write("/output/file.txt") is True


class TestPathNormalization:
    """Test path normalization edge cases."""
    
    def test_backslash_conversion(self):
        """Backslashes should be converted to forward slashes."""
        vfs = VirtualFileSystem()
        # This should not raise
        path = vfs._normalize_virtual("\\templates\\file.txt")
        assert "/" in path
        assert "\\" not in path
    
    def test_missing_leading_slash(self):
        """Paths without leading slash should be normalized."""
        vfs = VirtualFileSystem()
        path = vfs._normalize_virtual("templates/file.txt")
        assert path.startswith("/")
    
    def test_double_slashes(self):
        """Double slashes should be collapsed."""
        vfs = VirtualFileSystem()
        path = vfs._normalize_virtual("/templates//subfolder///file.txt")
        assert "//" not in path

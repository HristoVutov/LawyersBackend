"""
Integration tests for the agent system.

Tests:
- Agent initialization
- Tool execution
- Orchestrator delegation
- Backend persistence
"""
import pytest
import asyncio
from pathlib import Path


# ============================================
# BACKEND TESTS
# ============================================

class TestStateBackend:
    """Test ephemeral state backend."""
    
    def test_write_and_read(self):
        from app.middleware import StateBackend
        
        backend = StateBackend("test")
        backend.write_file("/notes.md", "# Test Notes")
        
        assert backend.exists("/notes.md")
        assert backend.read_file("/notes.md") == "# Test Notes"
    
    def test_list_files(self):
        from app.middleware import StateBackend
        
        backend = StateBackend("test")
        backend.write_file("/dir/file1.txt", "content1")
        backend.write_file("/dir/file2.txt", "content2")
        backend.write_file("/root.txt", "root content")
        
        files = backend.list_files("/dir/")
        assert "file1.txt" in files
        assert "file2.txt" in files
    
    def test_delete(self):
        from app.middleware import StateBackend
        
        backend = StateBackend("test")
        backend.write_file("/temp.txt", "temporary")
        assert backend.exists("/temp.txt")
        
        backend.delete_file("/temp.txt")
        assert not backend.exists("/temp.txt")


class TestStoreBackend:
    """Test persistent store backend."""
    
    def test_persistence(self, tmp_path):
        from app.middleware.backends import StoreBackend
        
        store_path = tmp_path / "test_store.json"
        
        # Write
        backend1 = StoreBackend(store_path)
        backend1.write_file("/memories/entity.md", "GALAXY OOD")
        
        # Read in new instance
        backend2 = StoreBackend(store_path)
        assert backend2.read_file("/memories/entity.md") == "GALAXY OOD"


class TestFilesystemBackend:
    """Test filesystem backend with sandboxing."""
    
    def test_list_real_files(self, tmp_path):
        from app.middleware.backends import FilesystemBackend
        
        # Create test files
        (tmp_path / "doc1.txt").write_text("content1")
        (tmp_path / "doc2.txt").write_text("content2")
        
        backend = FilesystemBackend(tmp_path)
        files = backend.list_files("/")
        
        assert "doc1.txt" in files
        assert "doc2.txt" in files
    
    def test_sandbox_escape_blocked(self, tmp_path):
        from app.middleware.backends import FilesystemBackend
        
        backend = FilesystemBackend(tmp_path, virtual_mode=True)
        
        # Attempt to escape sandbox
        with pytest.raises(ValueError, match="escapes sandbox"):
            backend._resolve_path("/../../../etc/passwd")


class TestCompositeBackend:
    """Test composite backend routing."""
    
    def test_routing(self, tmp_path):
        from app.middleware.backends import (
            CompositeBackend, StateBackend, StoreBackend, FilesystemBackend
        )
        
        composite = CompositeBackend(
            default=StateBackend("test"),
            routes={
                "/memories/": StoreBackend(tmp_path / "store.json"),
            }
        )
        
        # Default route
        composite.write_file("/scratch.txt", "ephemeral")
        assert composite.read_file("/scratch.txt") == "ephemeral"
        
        # Memories route
        composite.write_file("/memories/entity.md", "persistent")
        assert composite.read_file("/memories/entity.md") == "persistent"


# ============================================
# MIDDLEWARE TESTS
# ============================================

class TestTodoListMiddleware:
    """Test todo list middleware."""
    
    def test_context_injection(self):
        from app.middleware import (
            TodoListMiddleware, set_todo_conversation_id, 
            todo_store, TodoItem, get_todos_context
        )
        
        middleware = TodoListMiddleware()
        set_todo_conversation_id("test-conv")
        todo_store.set("test-conv", [
            TodoItem("Find documents", "in_progress"),
            TodoItem("Analyze data", "pending"),
        ])
        
        context = get_todos_context("test-conv")
        assert "[CURRENT TODO LIST]" in context
        assert "Find documents" in context


class TestRateLimitMiddleware:
    """Test rate limiting."""
    
    def test_global_limit(self):
        from app.middleware import RateLimitMiddleware, RateLimitConfig
        
        middleware = RateLimitMiddleware(RateLimitConfig(max_calls_per_minute=3))
        
        # First 3 calls should succeed
        for _ in range(3):
            allowed, _ = middleware.check("test")
            middleware.record("test")
            assert allowed
        
        # 4th call should be blocked
        allowed, msg = middleware.check("test")
        assert not allowed
        assert "Rate limit" in msg


class TestMaxIterationsMiddleware:
    """Test max iterations guard."""
    
    def test_iteration_limit(self):
        from app.middleware import MaxIterationsMiddleware
        
        middleware = MaxIterationsMiddleware(max_iterations=5, warning_threshold=3)
        
        # Increment 5 times and check after each
        for i in range(5):
            middleware.increment("test")
            count = middleware.get_count("test")
            allowed, msg = middleware.check("test")
            
            if count < 3:
                assert allowed
                assert msg is None, f"Expected no warning at count {count}"
            elif count < 5:
                assert allowed
                # Warning threshold hit
            else:
                # At limit
                assert not allowed


# ============================================
# AGENT TESTS
# ============================================

class TestAgentImports:
    """Test all agents can be imported."""
    
    def test_import_agents(self):
        from app.agents import (
            BaseAgent,
            DocumentAgent,
            ResearchAgent,
            DraftingAgent,
            TemplateAgent,
            OrchestratorAgent,
        )
        
        # Verify classes exist
        assert BaseAgent is not None
        assert DocumentAgent is not None
        assert OrchestratorAgent is not None


class TestAgentInstantiation:
    """Test agents can be instantiated (without API calls)."""
    
    def test_document_agent_init(self):
        from app.agents import DocumentAgent
        agent = DocumentAgent()
        assert agent.name == "document_agent"
        assert len(agent.tools) > 0
    
    def test_orchestrator_init(self):
        from app.agents import OrchestratorAgent
        agent = OrchestratorAgent()
        assert agent.name == "orchestrator"
        assert agent.max_iterations == 15  # Default from AgentConfig


# ============================================
# API TESTS
# ============================================

class TestHealthEndpoint:
    """Test health check endpoint."""
    
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


# Run with: python -m pytest tests/test_integration.py -v

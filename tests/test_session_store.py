"""
Tests for session store functionality.
"""
import pytest
from app.services.session_store import (
    SessionStore,
    mark_document_as_read,
    mark_document_in_context,
    is_document_read,
    is_document_in_context,
    cache_search_results,
    get_cached_search,
    remember_entity,
    find_entity,
    add_task,
    get_task_plan,
    update_task_status,
    get_next_task,
)


@pytest.fixture
def store():
    return SessionStore()


class TestDocumentTracking:
    async def test_mark_document_as_read(self, store):
        await mark_document_as_read(store, "conv-1", "test.pdf")
        assert await is_document_read(store, "conv-1", "test.pdf")
        assert not await is_document_read(store, "conv-1", "other.pdf")
    
    async def test_mark_document_in_context(self, store):
        await mark_document_in_context(store, "conv-1", "test.pdf")
        assert await is_document_in_context(store, "conv-1", "test.pdf")
    
    async def test_no_duplicates(self, store):
        await mark_document_as_read(store, "conv-1", "test.pdf")
        await mark_document_as_read(store, "conv-1", "test.pdf")
        state = await store.get_state("conv-1")
        assert state.documents_read.count("test.pdf") == 1


class TestSearchCaching:
    async def test_cache_and_retrieve(self, store):
        results = [{"file": "test.pdf", "score": 0.95}]
        await cache_search_results(store, "conv-1", "Galaxy properties", results)
        
        cached = await get_cached_search(store, "conv-1", "galaxy properties")
        assert cached == results
    
    async def test_cache_miss(self, store):
        cached = await get_cached_search(store, "conv-1", "nonexistent query")
        assert cached is None


class TestEntityMemory:
    async def test_remember_entity(self, store):
        await remember_entity(store, "conv-1", "GALAXY PROPERTY GROUP", {
            "type": "company",
            "bulstat": "115509755",
        })
        
        found = await find_entity(store, "conv-1", "galaxy")
        assert found is not None
        assert found["type"] == "company"


class TestTaskPlanning:
    async def test_add_and_get_task(self, store):
        await add_task(store, "conv-1", {
            "id": "t1",
            "description": "Find documents",
            "assigned_agent": "document_agent",
        })
        
        plan = await get_task_plan(store, "conv-1")
        assert len(plan["tasks"]) == 1
        assert plan["tasks"][0]["id"] == "t1"
    
    async def test_task_status_update(self, store):
        await add_task(store, "conv-1", {"id": "t1", "description": "Test"})
        await update_task_status(store, "conv-1", "t1", "done", "Completed successfully")
        
        plan = await get_task_plan(store, "conv-1")
        assert plan["tasks"][0]["status"] == "done"
        assert plan["tasks"][0]["result"] == "Completed successfully"
    
    async def test_get_next_task(self, store):
        await add_task(store, "conv-1", {"id": "t1", "description": "First"})
        await add_task(store, "conv-1", {"id": "t2", "description": "Second", "depends_on": ["t1"]})
        
        # Should get t1 first
        next_task = await get_next_task(store, "conv-1")
        assert next_task["id"] == "t1"
        
        # Complete t1
        await update_task_status(store, "conv-1", "t1", "done")
        
        # Now should get t2
        next_task = await get_next_task(store, "conv-1")
        assert next_task["id"] == "t2"

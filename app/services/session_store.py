"""
Session Store - Centralized memory for tracking session state across agent turns.

Uses a simple in-memory store to track:
- Documents added to context
- Documents read in full
- Last search queries
- Remembered entities

This prevents redundant document reads when users ask follow-up questions.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import time


@dataclass
class SearchCacheEntry:
    """Cached search result with TTL."""
    results: list[dict]
    timestamp: float
    ttl_ms: int = 300000  # 5 minutes default


@dataclass
class SessionState:
    """State for a single conversation session."""
    documents_in_context: list[str] = field(default_factory=list)
    documents_in_context: list[str] = field(default_factory=list)
    documents_read: list[str] = field(default_factory=list)
    document_content_cache: dict[str, str] = field(default_factory=dict)
    last_search_query: str | None = None
    last_search_results: list[dict] = field(default_factory=list)
    search_cache: dict[str, SearchCacheEntry] = field(default_factory=dict)
    entities: dict[str, dict] = field(default_factory=dict)
    task_plan: dict = field(default_factory=lambda: {"goal": "", "tasks": []})


class SessionStore:
    """
    In-memory session store for tracking state across agent turns.
    
    Compatible with LangGraph's InMemoryStore API pattern.
    """
    
    def __init__(self):
        self._store: dict[str, SessionState] = {}
    
    def _get_key(self, conversation_id: str) -> str:
        return f"session:{conversation_id}"
    
    async def get_state(self, conversation_id: str) -> SessionState:
        """Get session state for a conversation, creating if needed."""
        key = self._get_key(conversation_id)
        if key not in self._store:
            self._store[key] = SessionState()
        return self._store[key]
    
    async def update_state(self, conversation_id: str, **updates) -> SessionState:
        """Update session state with new values (merges lists, replaces scalars)."""
        state = await self.get_state(conversation_id)
        
        # Smart merge for document lists (add, avoid duplicates)
        if "documents_in_context" in updates:
            existing = set(state.documents_in_context)
            for doc in updates["documents_in_context"]:
                existing.add(doc)
            state.documents_in_context = list(existing)
        
        if "documents_read" in updates:
            existing = set(state.documents_read)
            for doc in updates["documents_read"]:
                existing.add(doc)
            state.documents_read = list(existing)

        if "document_content_cache" in updates:
            state.document_content_cache.update(updates["document_content_cache"])
        
        # Replace scalars
        if "last_search_query" in updates:
            state.last_search_query = updates["last_search_query"]
        
        if "last_search_results" in updates:
            state.last_search_results = updates["last_search_results"]
        
        if "search_cache" in updates:
            state.search_cache = updates["search_cache"]
        
        if "entities" in updates:
            state.entities = updates["entities"]
        
        if "task_plan" in updates:
            state.task_plan = updates["task_plan"]
        
        return state
    
    async def clear(self, conversation_id: str) -> None:
        """Clear state for a specific conversation."""
        key = self._get_key(conversation_id)
        if key in self._store:
            del self._store[key]
    
    async def clear_all(self) -> None:
        """Clear all sessions."""
        self._store.clear()


# ============================================
# DOCUMENT TRACKING
# ============================================

async def mark_document_as_read(store: SessionStore, conversation_id: str, file_name: str) -> SessionState:
    """Add a document to the 'read' list."""
    return await store.update_state(conversation_id, documents_read=[file_name])


async def mark_document_in_context(store: SessionStore, conversation_id: str, file_name: str) -> SessionState:
    """Add a document to the 'in context' list."""
    return await store.update_state(conversation_id, documents_in_context=[file_name])


async def is_document_read(store: SessionStore, conversation_id: str, file_name: str) -> bool:
    """Check if a document has already been read."""
    state = await store.get_state(conversation_id)
    return file_name in state.documents_read


async def is_document_in_context(store: SessionStore, conversation_id: str, file_name: str) -> bool:
    """Check if a document is already in context."""
    state = await store.get_state(conversation_id)
    return file_name in state.documents_in_context

async def cache_document_content(store: SessionStore, conversation_id: str, file_name: str, content: str) -> SessionState:
    """Cache the content of a document."""
    return await store.update_state(conversation_id, document_content_cache={file_name: content})

async def get_cached_document_content(store: SessionStore, conversation_id: str, file_name: str) -> str | None:
    """Get content of a cached document."""
    state = await store.get_state(conversation_id)
    return state.document_content_cache.get(file_name)


# ============================================
# SEARCH CACHING
# ============================================

def normalize_query(query: str) -> str:
    """Normalize search query for cache key."""
    return " ".join(query.lower().strip().split())


async def cache_search_results(
    store: SessionStore,
    conversation_id: str,
    query: str,
    results: list[dict],
    ttl_ms: int = 300000
) -> None:
    """Cache search results for a query."""
    state = await store.get_state(conversation_id)
    normalized = normalize_query(query)
    state.search_cache[normalized] = SearchCacheEntry(
        results=results,
        timestamp=time.time(),
        ttl_ms=ttl_ms
    )


async def get_cached_search(
    store: SessionStore,
    conversation_id: str,
    query: str
) -> list[dict] | None:
    """Get cached search results if they exist and are not expired."""
    state = await store.get_state(conversation_id)
    normalized = normalize_query(query)
    
    cached = state.search_cache.get(normalized)
    if not cached:
        return None
    
    # Check expiry
    age_ms = (time.time() - cached.timestamp) * 1000
    if age_ms > cached.ttl_ms:
        del state.search_cache[normalized]
        return None
    
    return cached.results


# ============================================
# ENTITY MEMORY
# ============================================

async def remember_entity(
    store: SessionStore,
    conversation_id: str,
    entity_name: str,
    entity_data: dict
) -> None:
    """Add or update an entity in memory."""
    state = await store.get_state(conversation_id)
    normalized_name = entity_name.upper().strip()
    
    if normalized_name in state.entities:
        # Merge with existing
        existing = state.entities[normalized_name]
        # Merge related_docs lists
        if "related_docs" in entity_data and "related_docs" in existing:
            entity_data["related_docs"] = list(set(
                existing.get("related_docs", []) + entity_data.get("related_docs", [])
            ))
        # Merge aliases
        if "aliases" in entity_data and "aliases" in existing:
            entity_data["aliases"] = list(set(
                existing.get("aliases", []) + entity_data.get("aliases", [])
            ))
        existing.update(entity_data)
        existing["last_updated"] = datetime.now().isoformat()
    else:
        state.entities[normalized_name] = {
            **entity_data,
            "name": entity_name,
            "first_seen": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }


async def find_entity(
    store: SessionStore,
    conversation_id: str,
    search_name: str
) -> dict | None:
    """Find an entity by name or alias."""
    state = await store.get_state(conversation_id)
    normalized = search_name.upper().strip()
    
    # Direct match
    if normalized in state.entities:
        return {"name": normalized, **state.entities[normalized]}
    
    # Search in aliases
    for name, data in state.entities.items():
        aliases = data.get("aliases", [])
        for alias in aliases:
            if normalized in alias.upper() or alias.upper() in normalized:
                return {"name": name, **data}
    
    # Partial match on name
    for name, data in state.entities.items():
        if normalized in name or name in normalized:
            return {"name": name, **data}
    
    return None


async def get_all_entities(store: SessionStore, conversation_id: str) -> dict:
    """Get all remembered entities."""
    state = await store.get_state(conversation_id)
    return state.entities


# ============================================
# TASK PLANNING (Deep Agents Pattern)
# ============================================

async def set_task_goal(store: SessionStore, conversation_id: str, goal: str) -> dict:
    """Set the overall goal for the task plan."""
    state = await store.get_state(conversation_id)
    state.task_plan["goal"] = goal
    return state.task_plan


async def add_task(store: SessionStore, conversation_id: str, task: dict) -> dict:
    """Add a task to the orchestrator's plan."""
    state = await store.get_state(conversation_id)
    task_plan = state.task_plan
    
    # Set defaults
    task.setdefault("status", "pending")
    task.setdefault("created_at", datetime.now().isoformat())
    
    # Update or add
    existing_idx = next(
        (i for i, t in enumerate(task_plan["tasks"]) if t.get("id") == task.get("id")),
        None
    )
    if existing_idx is not None:
        task_plan["tasks"][existing_idx].update(task)
    else:
        task_plan["tasks"].append(task)
    
    return task_plan


async def update_task_status(
    store: SessionStore,
    conversation_id: str,
    task_id: str,
    status: str,
    result: str | None = None
) -> dict:
    """Update a task's status."""
    state = await store.get_state(conversation_id)
    
    for task in state.task_plan["tasks"]:
        if task.get("id") == task_id:
            task["status"] = status
            task["updated_at"] = datetime.now().isoformat()
            if result:
                task["result"] = result
            break
    
    return state.task_plan


async def get_task_plan(store: SessionStore, conversation_id: str) -> dict:
    """Get the full task plan for a conversation."""
    state = await store.get_state(conversation_id)
    return state.task_plan


async def clear_task_plan(store: SessionStore, conversation_id: str) -> dict:
    """Clear/reset the task plan."""
    state = await store.get_state(conversation_id)
    state.task_plan = {"goal": "", "tasks": []}
    return state.task_plan


async def get_next_task(store: SessionStore, conversation_id: str) -> dict | None:
    """Get next pending task that has all dependencies satisfied."""
    state = await store.get_state(conversation_id)
    task_plan = state.task_plan
    
    for task in task_plan["tasks"]:
        if task.get("status") != "pending":
            continue
        
        # Check dependencies
        depends_on = task.get("depends_on", [])
        all_done = all(
            any(t.get("id") == dep_id and t.get("status") == "done" for t in task_plan["tasks"])
            for dep_id in depends_on
        )
        
        if all_done or not depends_on:
            return task
    
    return None


# ============================================
# SINGLETON INSTANCE
# ============================================

# Global session store instance
session_store = SessionStore()

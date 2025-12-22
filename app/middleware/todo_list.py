"""
TODO List Middleware - Deep Agents Pattern.

Implements the LangChain Deep Agents TodoListMiddleware pattern:
- Adds a `write_todos` tool for agents to manage their task list
- Injects current todos into system prompt so agent sees its pending work
- Persists todos across conversation turns

Reference: https://docs.langchain.com/oss/python/deepagents/middleware#to-do-list-middleware
"""
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from app.middleware.prompts.todo_list_prompt import TODO_LIST_SYSTEM_PROMPT


@dataclass
class TodoItem:
    """A single todo item."""
    task: str
    status: str = "pending"  # pending, in_progress, completed
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class TodoStore:
    """Per-conversation todo storage."""
    
    def __init__(self):
        self._stores: dict[str, list[TodoItem]] = {}
    
    def get(self, conversation_id: str) -> list[TodoItem]:
        """Get todos for a conversation."""
        if conversation_id not in self._stores:
            self._stores[conversation_id] = []
        return self._stores[conversation_id]
    
    def set(self, conversation_id: str, todos: list[TodoItem]) -> None:
        """Replace todos for a conversation."""
        self._stores[conversation_id] = todos
    
    def clear(self, conversation_id: str) -> None:
        """Clear todos for a conversation."""
        self._stores[conversation_id] = []


# Global todo store instance
todo_store = TodoStore()

# Current conversation ID (set before tool execution)
_current_conversation_id = "default"


def set_todo_conversation_id(conversation_id: str) -> None:
    """Set the current conversation ID for todo operations."""
    global _current_conversation_id
    _current_conversation_id = conversation_id


def get_todos_context(conversation_id: str | None = None) -> str:
    """
    Get current todos formatted for context injection.
    
    This string is prepended to agent messages so they can SEE existing todos.
    """
    cid = conversation_id or _current_conversation_id
    todos = todo_store.get(cid)
    
    if not todos:
        return ""
    
    lines = []
    for i, item in enumerate(todos):
        if item.status == "completed":
            status = "✓"
        elif item.status == "in_progress":
            status = "→"
        else:
            status = "○"
        lines.append(f"  [{status}] {i + 1}. {item.task}")
    
    return f"[CURRENT TODO LIST]\n" + "\n".join(lines) + "\n\n"


# Pydantic models for tool schema
class TodoInput(BaseModel):
    """Input schema for write_todos tool."""
    action: str = Field(description="Action: 'write', 'update', 'read', or 'clear'")
    todos: list[dict] | None = Field(
        default=None,
        description="Array of {task, status?} objects for 'write' action"
    )
    todo_index: int | None = Field(
        default=None,
        description="Index of todo to update (0-based) for 'update' action"
    )
    status: str | None = Field(
        default=None,
        description="New status ('pending', 'in_progress', 'completed') for 'update' action"
    )


@tool
def write_todos(
    action: str,
    todos: list[dict] | None = None,
    todo_index: int | None = None,
    status: str | None = None,
    config: RunnableConfig = None
) -> str:
    """
    Manage your task list. The current state of your todos is shown in context.
    
    Actions:
    - "write": Replace entire todo list with new items. Provide 'todos' array.
    - "update": Update a specific item's status. Provide 'todo_index' and 'status'.
    - "read": Get current todo list.
    - "clear": Remove all todos.
    
    Always check [CURRENT TODO LIST] in context before writing to avoid duplicates.
    """
    # Resolve conversation ID: Config (Best) -> Global (Fallback)
    conversation_id = _current_conversation_id
    if config and "configurable" in config and "thread_id" in config["configurable"]:
        conversation_id = config["configurable"]["thread_id"]
    
    store = todo_store.get(conversation_id)
    
    try:
        if action == "write":
            if not todos:
                return "Error: 'todos' array required for write action"
            
            new_todos = [
                TodoItem(
                    task=t.get("task", str(t)) if isinstance(t, dict) else str(t),
                    status=t.get("status", "pending") if isinstance(t, dict) else "pending"
                )
                for t in todos
            ]
            todo_store.set(conversation_id, new_todos)
            print(f"[write_todos] Wrote {len(todos)} items for {conversation_id}")
            return f"✅ Todo list updated with {len(todos)} items:\n{get_todos_context(conversation_id)}"
        
        elif action == "update":
            if todo_index is None or todo_index < 0 or todo_index >= len(store):
                return f"Error: Invalid todo_index {todo_index}. List has {len(store)} items (0-indexed)."
            
            store[todo_index].status = status or store[todo_index].status
            print(f"[write_todos] Updated item {todo_index} to {status}")
            return f"✅ Updated item {todo_index + 1}: \"{store[todo_index].task}\" → {store[todo_index].status}\n{get_todos_context(conversation_id)}"
        
        elif action == "read":
            if not store:
                return "Todo list is empty."
            return get_todos_context(conversation_id)
        
        elif action == "clear":
            todo_store.clear(conversation_id)
            print(f"[write_todos] Cleared todo list for {conversation_id}")
            return "✅ Todo list cleared."
        
        else:
            return f"Unknown action: {action}. Use: write, update, read, or clear"
    
    except Exception as e:
        return f"Error: {str(e)}"


class TodoListMiddleware:
    """
    Deep Agents TodoListMiddleware pattern.
    
    Usage:
        middleware = TodoListMiddleware()
        
        # Before agent call, inject todos into message:
        message = middleware.before(message, conversation_id)
        
        # The write_todos tool is available in middleware.tools
    """
    
    DEFAULT_SYSTEM_PROMPT = TODO_LIST_SYSTEM_PROMPT
    
    def __init__(self, system_prompt: str | None = None):
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.tools = [write_todos]
    
    def before(self, message: str, conversation_id: str) -> str:
        """
        Called before each agent invocation.
        Injects current todo state into the message.
        """
        set_todo_conversation_id(conversation_id)
        todos_context = get_todos_context(conversation_id)
        
        if todos_context:
            return todos_context + message
        return message
    
    def get_system_prompt_addition(self) -> str:
        """Get the system prompt addition for TODO management."""
        return self.system_prompt

from typing import Any, Dict
from pydantic import BaseModel, Field

class RemoteToolCall(BaseModel):
    """Event sent from Server to UI to request local tool execution."""
    type: str = "local_tool_call"
    call_id: str
    tool_name: str
    args: Dict[str, Any]

class RemoteToolResult(BaseModel):
    """Event sent from UI back to Server with local tool result."""
    type: str = "local_tool_result"
    call_id: str
    result: Any
    error: str | None = None

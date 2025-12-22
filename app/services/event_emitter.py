"""
Event Emitter Service - Centralized event streaming for agent workflows.

Provides a consistent interface for emitting chronological events to the UI,
including agent thinking, tool calls, step transitions, and content streaming.
"""
import asyncio
from datetime import datetime
from typing import AsyncIterator, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class EventType(str, Enum):
    """Types of events that can be emitted."""
    AGENT_START = "agent-start"
    AGENT_END = "agent-end"
    STEP = "step"
    THINKING = "thinking"          # Streaming thinking content
    THINKING_COMPLETE = "thinking-complete"  # Thinking phase done
    TOOL_CALL = "tool-call"
    TOOL_RESULT = "tool-result"
    CONTENT = "content"
    CONTEXT_UPDATE = "context-update"
    ERROR = "error"
    DONE = "done"


@dataclass
class AgentEvent:
    """A single event in the agent workflow."""
    type: EventType
    agent: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    step: Optional[str] = None
    text: Optional[str] = None
    tool: Optional[str] = None
    args: Optional[dict] = None
    result: Optional[Any] = None
    task: Optional[str] = None
    description: Optional[str] = None
    error: Optional[str] = None
    files: Optional[list] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = {
            "type": self.type.value if isinstance(self.type, EventType) else self.type,
            "agent": self.agent,
            "timestamp": self.timestamp,
        }
        # Only include non-None optional fields
        if self.step is not None:
            data["step"] = self.step
        if self.text is not None:
            data["text"] = self.text
        if self.tool is not None:
            data["tool"] = self.tool
        if self.args is not None:
            data["args"] = self.args
        if self.result is not None:
            data["result"] = self.result
        if self.task is not None:
            data["task"] = self.task
        if self.description is not None:
            data["description"] = self.description
        if self.error is not None:
            data["error"] = self.error
        if self.files is not None:
            data["files"] = self.files
        return data


class EventEmitter:
    """
    Centralized event emission for agent workflows.
    
    Uses an async queue to buffer events, allowing multiple agents
    to emit events that are consumed by a single stream.
    """
    
    def __init__(self):
        self._queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        self._closed = False
    
    async def emit(self, event: AgentEvent) -> None:
        """Emit an event to the stream."""
        if not self._closed:
            await self._queue.put(event)
    
    async def emit_raw(self, data: dict) -> None:
        """Emit a raw dictionary event (for compatibility with existing code)."""
        if not self._closed:
            # Convert to AgentEvent-like structure
            event_type = data.get("type", "content")
            agent = data.get("agent", "unknown")
            
            event = AgentEvent(
                type=event_type,
                agent=agent,
                step=data.get("step"),
                text=data.get("text"),
                tool=data.get("tool") or data.get("toolName"),
                args=data.get("args"),
                result=data.get("result"),
                task=data.get("task"),
                description=data.get("description"),
                error=data.get("error"),
                files=data.get("files"),
            )
            await self._queue.put(event)
    
    async def agent_start(self, agent: str, task: str) -> None:
        """Emit agent start event."""
        await self.emit(AgentEvent(
            type=EventType.AGENT_START,
            agent=agent,
            task=task,
        ))
    
    async def agent_end(self, agent: str) -> None:
        """Emit agent completion event."""
        await self.emit(AgentEvent(
            type=EventType.AGENT_END,
            agent=agent,
        ))
    
    async def step(self, agent: str, step: str, description: str = None) -> None:
        """Emit step/node transition event."""
        await self.emit(AgentEvent(
            type=EventType.STEP,
            agent=agent,
            step=step,
            description=description,
        ))
    
    async def thinking(self, agent: str, text: str, step: str = None) -> None:
        """Emit streaming thinking content."""
        await self.emit(AgentEvent(
            type=EventType.THINKING,
            agent=agent,
            text=text,
            step=step,
        ))
    
    async def thinking_complete(self, agent: str, step: str = None) -> None:
        """Emit thinking complete event."""
        await self.emit(AgentEvent(
            type=EventType.THINKING_COMPLETE,
            agent=agent,
            step=step,
        ))
    
    async def tool_call(self, agent: str, tool: str, args: dict, step: str = None) -> None:
        """Emit tool invocation event."""
        await self.emit(AgentEvent(
            type=EventType.TOOL_CALL,
            agent=agent,
            tool=tool,
            args=args,
            step=step,
        ))
    
    async def tool_result(self, agent: str, tool: str, result: Any, step: str = None) -> None:
        """Emit tool completion event."""
        # Truncate large results
        if isinstance(result, str) and len(result) > 500:
            result = result[:500] + "..."
        await self.emit(AgentEvent(
            type=EventType.TOOL_RESULT,
            agent=agent,
            tool=tool,
            result=result,
            step=step,
        ))
    
    async def content(self, agent: str, text: str, step: str = None) -> None:
        """Emit streaming content event."""
        await self.emit(AgentEvent(
            type=EventType.CONTENT,
            agent=agent,
            text=text,
            step=step,
        ))
    
    async def error(self, agent: str, error: str) -> None:
        """Emit error event."""
        await self.emit(AgentEvent(
            type=EventType.ERROR,
            agent=agent,
            error=error,
        ))
    
    async def done(self, agent: str = "orchestrator") -> None:
        """Emit done event and close the stream."""
        await self.emit(AgentEvent(
            type=EventType.DONE,
            agent=agent,
        ))
        self._closed = True
        await self._queue.put(None)  # Signal end of stream
    
    async def stream(self) -> AsyncIterator[dict]:
        """Iterate over events as they are emitted."""
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event.to_dict()
    
    def close(self) -> None:
        """Close the emitter without emitting done event."""
        self._closed = True
        # Put None to unblock any waiting consumers
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

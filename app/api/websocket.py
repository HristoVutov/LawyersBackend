"""
WebSocket API for streaming agent responses.

Provides real-time bidirectional communication for the chat interface.
Connects to OrchestratorAgent for multi-agent coordination.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import json
import traceback
import uuid
import asyncio

from app.config import get_settings
from app.services.conversation_logger import get_conversation_logger, log_ws_event, print_and_log
from app.remote.manager import remote_tool_manager
from app.remote.protocol import RemoteToolResult
from app.remote.agent_bridge import patch_all_agents
from app.api.manager import manager

router = APIRouter()


class ChatMessage(BaseModel):
    """Incoming chat message from client."""
    message: str
    thread_id: str = "default"
    context_files: list[str] = []
    model: str | None = None


# manager is imported from .manager to avoid circular imports

# Lazy load to avoid circular imports
_orchestrator = None


def get_orchestrator():
    """Get or create the orchestrator agent."""
    global _orchestrator
    
    if _orchestrator is None:
        from app.agents import OrchestratorAgent, initialize_agent_registry
        
        # Initialize all agents in registry
        initialize_agent_registry()
        
        # Create orchestrator
        _orchestrator = OrchestratorAgent()
        print("✅ Orchestrator and agent registry initialized")
    
    return _orchestrator


@router.websocket("/chat/{client_id}")
async def chat_websocket(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for streaming chat responses.
    
    Protocol:
    - Client sends: {"message": "...", "thread_id": "optional", "context_files": [...]}
    - If thread_id is not provided, server generates a new UUID and sends:
      {"type": "thread_started", "thread_id": "uuid"}
    - Client should save this thread_id and include it in subsequent messages
      to continue the same conversation
    - Server streams: {"type": "content|tool-call|tool-result|done|error", ...}
    - Client can send {"type": "local_tool_result", ...} at any time
    """
    await manager.connect(websocket, client_id)
    
    # Queue for processing chat messages sequentially
    chat_queue = asyncio.Queue()
    
    async def process_chat_message():
        """Worker to process chat messages from the queue."""
        while True:
            data = await chat_queue.get()
            try:
                message = data.get("message", "")
                context_files = data.get("context_files", [])
                
                # Handle thread_id: generate new UUID if not provided (new conversation)
                provided_thread_id = data.get("thread_id")
                is_new_thread = not provided_thread_id or provided_thread_id in ("", "default", None)
                
                if is_new_thread:
                    thread_id = str(uuid.uuid4())
                    # Notify client of the new thread_id so they can use it for subsequent messages
                    await manager.send_json(client_id, {
                        "type": "thread_started",
                        "thread_id": thread_id,
                    }, thread_id=thread_id)
                    print_and_log(f"🆕 [{client_id}] New conversation started: {thread_id}", thread_id)
                else:
                    thread_id = provided_thread_id
                
                # Log incoming message now that we have thread_id
                log_ws_event(client_id, data, direction="IN", thread_id=thread_id)
                
                # Register this thread to this client for remote tool routing
                remote_tool_manager.register_client(thread_id, client_id)
                
                msg_preview = message[:50] + "..." if len(message) > 50 else message
                print_and_log(f"📨 [{client_id}] Received: {msg_preview} (thread: {thread_id[:8]}...)", thread_id)
                
                # Processing logic
                settings = get_settings()
                orchestrator = get_orchestrator() # Initialize first!
                
                if settings.remote_tool_mode:
                    patch_all_agents(thread_id, is_remote=True)
                
                # Stream response from orchestrator
                async for chunk in orchestrator.stream(
                    message=message,
                    thread_id=thread_id,
                    context_files=context_files,
                    project_context=data.get("project_context"),
                    model_name=data.get("model")
                ):
                    # Forward chunk to client with thread_id for logging
                    await manager.send_json(client_id, chunk, thread_id=thread_id)
                    
                    # Log important events
                    if chunk.get("type") == "tool-call":
                        tools = [tc.get("name") for tc in chunk.get("toolCalls", [])]
                        print_and_log(f"🔧 [{client_id}] Tool calls: {tools}", thread_id)
                    elif chunk.get("type") == "error":
                        print_and_log(f"❌ [{client_id}] Agent error: {chunk.get('error')}", thread_id)
                
                # Send done signal (agent also sends this, but ensure it's sent)
                await manager.send_json(client_id, {"type": "done"}, thread_id=thread_id)
                
            except Exception as agent_error:
                # Handle agent errors gracefully
                error_msg = str(agent_error)
                print_and_log(f"❌ [{client_id}] Agent error: {error_msg}", thread_id if 'thread_id' in locals() else "unknown")
                traceback.print_exc()
                
                if 'thread_id' in locals():
                    await manager.send_json(client_id, {
                        "type": "error",
                        "error": error_msg,
                        "agent": "orchestrator",
                    }, thread_id=thread_id)
                    await manager.send_json(client_id, {"type": "done"}, thread_id=thread_id)
            finally:
                chat_queue.task_done()

    # Start the worker task
    worker_task = asyncio.create_task(process_chat_message())

    try:
        while True:
            # Receive message from client (BLOCKING call)
            # This loop must stay responsive to receive incoming tool results
            data = await websocket.receive_json()
            
            # Check if this is a local tool result from the UI
            if data.get("type") == "local_tool_result":
                try:
                    result = RemoteToolResult(**data)
                    remote_tool_manager.resolve_tool_call(result)
                    # We don't log the thread_id here comfortably as we don't have it easily
                    # But resolve_tool_call logs internally if needed or we can log generically
                    print_and_log(f"📥 [{client_id}] Resolved tool call: {result.call_id}", "system")
                    continue # Ready for next message immediately
                except Exception as e:
                    print_and_log(f"❌ [{client_id}] Error parsing tool result: {e}", "system")
                    continue
            
            # If it's not a tool result, it's a chat message/command
            # Offload to queue to avoid blocking this receive loop
            await chat_queue.put(data)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        worker_task.cancel()
    except Exception as e:
        print(f"❌ WebSocket error for {client_id}: {e}")
        traceback.print_exc()
        manager.disconnect(client_id)
        worker_task.cancel()


@router.get("/connections")
async def list_connections():
    """List active WebSocket connections."""
    return {
        "active_connections": list(manager.active_connections.keys()),
        "count": len(manager.active_connections),
    }

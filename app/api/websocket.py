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

from app.services.conversation_logger import get_conversation_logger, log_ws_event, print_and_log

router = APIRouter()


class ChatMessage(BaseModel):
    """Incoming chat message from client."""
    message: str
    thread_id: str = "default"
    context_files: list[str] = []
    model: str | None = None


class ConnectionManager:
    """Manages active WebSocket connections."""
    
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        print(f"🔌 Client connected: {client_id}")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            print(f"🔌 Client disconnected: {client_id}")
    
    async def send_json(self, client_id: str, data: dict, thread_id: str | None = None):
        if client_id in self.active_connections:
            try:
                # Use a custom generic default for serializing non-standard objects
                def custom_serializer(obj):
                    if hasattr(obj, "dict"):
                        return obj.dict()
                    if hasattr(obj, "to_json"):
                        return obj.to_json()
                    return str(obj)

                # Dump to string first to use custom serializer, then send as text
                # We use send_text because send_json doesn't support custom encoders
                json_str = json.dumps(data, default=custom_serializer, ensure_ascii=False)
                await self.active_connections[client_id].send_text(json_str)
                
                # Log outgoing WebSocket event
                log_ws_event(client_id, data, direction="OUT", thread_id=thread_id)
            except RuntimeError as e:
                # Client disconnected mid-stream - silently clean up
                if "close message" in str(e):
                    self.disconnect(client_id)
                else:
                    print(f"❌ Error sending to {client_id}: {e}")
            except Exception as e:
                print(f"❌ Error sending to {client_id}: {e}")
    
    def is_connected(self, client_id: str) -> bool:
        return client_id in self.active_connections


manager = ConnectionManager()

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
    """
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
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
            
            msg_preview = message[:50] + "..." if len(message) > 50 else message
            print_and_log(f"📨 [{client_id}] Received: {msg_preview} (thread: {thread_id[:8]}...)", thread_id)
            
            try:
                # Get orchestrator (lazy init)
                orchestrator = get_orchestrator()
                
                # Stream response from orchestrator
                async for chunk in orchestrator.stream(
                    message=message,
                    thread_id=thread_id,
                    context_files=context_files,
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
                print_and_log(f"❌ [{client_id}] Agent error: {error_msg}", thread_id)
                traceback.print_exc()
                
                await manager.send_json(client_id, {
                    "type": "error",
                    "error": error_msg,
                    "agent": "orchestrator",
                }, thread_id=thread_id)
                await manager.send_json(client_id, {"type": "done"}, thread_id=thread_id)
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        print(f"❌ WebSocket error for {client_id}: {e}")
        traceback.print_exc()
        manager.disconnect(client_id)


@router.get("/connections")
async def list_connections():
    """List active WebSocket connections."""
    return {
        "active_connections": list(manager.active_connections.keys()),
        "count": len(manager.active_connections),
    }

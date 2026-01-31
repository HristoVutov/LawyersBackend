import asyncio
import uuid
from typing import Any, Dict, Optional
from app.api.manager import manager as ws_manager
from app.remote.protocol import RemoteToolCall, RemoteToolResult
from app.services.conversation_logger import print_and_log

class RemoteToolManager:
    """
    Manages the lifecycle of remote tool calls.
    Allows Server-side agents to call tools that execute on the Client-side (UI).
    """
    
    def __init__(self):
        # Stores pending futures: {call_id: asyncio.Future}
        self._pending_calls: Dict[str, asyncio.Future] = {}
        # Stores mapping: {thread_id: client_id}
        self._thread_to_client: Dict[str, str] = {}

    def register_client(self, thread_id: str, client_id: str):
        """Register which client is handling which thread."""
        self._thread_to_client[thread_id] = client_id

    async def call_remote_tool(self, thread_id: str, tool_name: str, args: Dict[str, Any]) -> Any:
        """
        Sends a tool call to the client and waits for the result.
        """
        client_id = self._thread_to_client.get(thread_id)
        if not client_id or not ws_manager.is_connected(client_id):
            raise ConnectionError(f"Client for thread {thread_id} is not connected or registered.")

        call_id = str(uuid.uuid4())
        future = asyncio.Future()
        self._pending_calls[call_id] = future

        # Prepare payload
        payload = RemoteToolCall(
            call_id=call_id,
            tool_name=tool_name,
            args=args
        ).dict()

        print_and_log(f"📡 [RemoteTool] Sending tool call {tool_name} to {client_id}", thread_id)
        
        # Send via WebSocket
        await ws_manager.send_json(client_id, payload, thread_id=thread_id)

        try:
            # Wait for result (with timeout)
            result: RemoteToolResult = await asyncio.wait_for(future, timeout=60.0)
            
            if result.error:
                raise RuntimeError(f"Remote tool '{tool_name}' failed: {result.error}")
            
            return result.result
        except asyncio.TimeoutError:
            raise TimeoutError(f"Remote tool '{tool_name}' timed out after 60s.")
        finally:
            # Clean up
            if call_id in self._pending_calls:
                del self._pending_calls[call_id]

    def resolve_tool_call(self, result: RemoteToolResult):
        """
        Called when a TOOL_RESULT event arrives via WebSocket.
        """
        future = self._pending_calls.get(result.call_id)
        if future and not future.done():
            future.set_result(result)

# Global instances
remote_tool_manager = RemoteToolManager()

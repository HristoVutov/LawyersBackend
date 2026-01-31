import json
from fastapi import WebSocket
from app.services.conversation_logger import log_ws_event

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

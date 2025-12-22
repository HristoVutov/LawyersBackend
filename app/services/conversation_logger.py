"""
Conversation Logger Service.

Logs terminal output and WebSocket events to files in Documents/LawyersAI/conversations.
Each conversation (thread_id) gets its own folder:
- conversations/<thread_id>/terminal_log.txt - Terminal/console output for this conversation
- conversations/<thread_id>/websocket_events.txt - WebSocket events for this conversation
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any
import threading
from contextvars import ContextVar

# Context variable for the current thread_id (conversation)
# This allows agent code to log without explicitly passing thread_id
_current_thread_id: ContextVar[str | None] = ContextVar("current_thread_id", default=None)


def set_current_thread(thread_id: str | None):
    """Set the current thread_id for logging context."""
    _current_thread_id.set(thread_id)


def get_current_thread() -> str | None:
    """Get the current thread_id from logging context."""
    return _current_thread_id.get()


class ConversationLogger:
    """
    Logger that writes terminal output and WebSocket events
    to separate files in Documents/LawyersAI/conversations/<thread_id>/.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        
        # Base directory for all conversations
        self.base_path = Path.home() / "Documents" / "LawyersAI" / "conversations"
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Session ID for general/startup logs (before any conversation starts)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # General terminal log for startup/global messages
        self.general_terminal_log = self.base_path / f"server_log_{self.session_id}.txt"
        self._write_header(self.general_terminal_log, "SERVER LOG", self.session_id)
        
        # Track initialized conversation folders
        self._initialized_conversations: set[str] = set()
        
        print(f"📝 Conversation logging initialized:")
        print(f"   Base path: {self.base_path}")
        print(f"   Server log: {self.general_terminal_log}")
    
    def _get_conversation_path(self, thread_id: str) -> Path:
        """Get the folder path for a conversation."""
        return self.base_path / thread_id
    
    def _ensure_conversation_folder(self, thread_id: str):
        """Ensure conversation folder exists and has headers."""
        if thread_id in self._initialized_conversations:
            return
            
        conv_path = self._get_conversation_path(thread_id)
        conv_path.mkdir(parents=True, exist_ok=True)
        
        # Create log files with headers
        terminal_log = conv_path / "terminal_log.txt"
        websocket_log = conv_path / "websocket_events.txt"
        
        if not terminal_log.exists():
            self._write_header(terminal_log, "TERMINAL LOG", thread_id)
        if not websocket_log.exists():
            self._write_header(websocket_log, "WEBSOCKET EVENTS LOG", thread_id)
        
        self._initialized_conversations.add(thread_id)
    
    def _write_header(self, path: Path, title: str, conversation_id: str):
        """Write header to log file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{'='*60}\n")
            f.write(f"{title}\n")
            f.write(f"Conversation: {conversation_id}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n\n")
    
    def log_terminal(self, message: str, thread_id: str | None = None):
        """
        Log a terminal/console message.
        
        If thread_id is provided, logs to that conversation's folder.
        Otherwise logs to the general server log.
        """
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_line = f"[{timestamp}] {message}\n"
        
        if thread_id:
            self._ensure_conversation_folder(thread_id)
            log_path = self._get_conversation_path(thread_id) / "terminal_log.txt"
        else:
            log_path = self.general_terminal_log
            
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_line)
    
    def log_websocket_event(
        self,
        client_id: str,
        event_type: str,
        data: dict[str, Any],
        direction: str = "OUT",  # "OUT" for server->client, "IN" for client->server
        thread_id: str | None = None
    ):
        """
        Log a WebSocket event.
        
        Args:
            client_id: The client connection ID
            event_type: Type of event (content, tool-call, done, error, etc.)
            data: The full event data
            direction: "OUT" for outgoing (server->client), "IN" for incoming
            thread_id: The conversation thread ID for folder organization
        """
        # Try to extract thread_id from data if not provided
        if not thread_id:
            thread_id = data.get("thread_id")
        
        if not thread_id:
            # Log to general server log if no thread_id
            self.log_terminal(f"[WS {direction}] [{client_id}] {event_type}: {data}")
            return
            
        self._ensure_conversation_folder(thread_id)
        
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        arrow = "→" if direction == "OUT" else "←"
        
        log_path = self._get_conversation_path(thread_id) / "websocket_events.txt"
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n[{timestamp}] [{client_id}] {arrow} {event_type.upper()}\n")
            
            # Format data nicely
            try:
                formatted = json.dumps(data, indent=2, ensure_ascii=False, default=str)
                for line in formatted.split("\n"):
                    f.write(f"    {line}\n")
            except Exception:
                f.write(f"    {data}\n")
    
    def log_websocket_incoming(self, client_id: str, data: dict[str, Any], thread_id: str | None = None):
        """Log an incoming WebSocket message from client."""
        event_type = "message"
        self.log_websocket_event(client_id, event_type, data, direction="IN", thread_id=thread_id)
    
    def log_websocket_outgoing(self, client_id: str, data: dict[str, Any], thread_id: str | None = None):
        """Log an outgoing WebSocket event to client."""
        event_type = data.get("type", "unknown")
        self.log_websocket_event(client_id, event_type, data, direction="OUT", thread_id=thread_id)


# Global instance getter
def get_conversation_logger() -> ConversationLogger:
    """Get the singleton conversation logger instance."""
    return ConversationLogger()


# Convenience functions
def log_terminal(message: str, thread_id: str | None = None):
    """Log a terminal message."""
    get_conversation_logger().log_terminal(message, thread_id)


def log_ws_event(client_id: str, data: dict[str, Any], direction: str = "OUT", thread_id: str | None = None):
    """Log a WebSocket event."""
    if direction == "IN":
        get_conversation_logger().log_websocket_incoming(client_id, data, thread_id)
    else:
        get_conversation_logger().log_websocket_outgoing(client_id, data, thread_id)


def print_and_log(message: str, thread_id: str | None = None):
    """Print to console AND log to terminal file.
    
    Use this instead of plain print() when you want the message
    to appear in both the console and the conversation's terminal_log.txt.
    
    If thread_id is not provided, tries to use the current thread context.
    """
    print(message)
    # Use provided thread_id, or fall back to context variable
    effective_thread_id = thread_id or get_current_thread()
    get_conversation_logger().log_terminal(message, effective_thread_id)


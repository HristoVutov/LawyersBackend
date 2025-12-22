"""
Rate Limiter Middleware - Prevents excessive API calls.

Provides:
- Global rate limiting per conversation
- Per-tool rate limiting
- Backoff on rate limit hits
"""
import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    max_calls_per_minute: int = 60
    max_calls_per_tool: int = 20
    backoff_seconds: float = 1.0


class RateLimiter:
    """
    Simple rate limiter for API calls.
    """
    
    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
        self._call_timestamps: dict[str, list[float]] = {}
        self._tool_counts: dict[str, dict[str, int]] = {}
    
    def _get_timestamps(self, conversation_id: str) -> list[float]:
        if conversation_id not in self._call_timestamps:
            self._call_timestamps[conversation_id] = []
        return self._call_timestamps[conversation_id]
    
    def _get_tool_count(self, conversation_id: str, tool_name: str) -> int:
        if conversation_id not in self._tool_counts:
            self._tool_counts[conversation_id] = {}
        return self._tool_counts[conversation_id].get(tool_name, 0)
    
    def _increment_tool_count(self, conversation_id: str, tool_name: str) -> None:
        if conversation_id not in self._tool_counts:
            self._tool_counts[conversation_id] = {}
        self._tool_counts[conversation_id][tool_name] = self._get_tool_count(conversation_id, tool_name) + 1
    
    def check_rate_limit(self, conversation_id: str, tool_name: str | None = None) -> tuple[bool, str]:
        """
        Check if a call is allowed.
        
        Returns:
            (allowed, message) tuple
        """
        now = time.time()
        timestamps = self._get_timestamps(conversation_id)
        
        # Clean old timestamps (older than 1 minute)
        cutoff = now - 60
        timestamps = [t for t in timestamps if t > cutoff]
        self._call_timestamps[conversation_id] = timestamps
        
        # Check global rate limit
        if len(timestamps) >= self.config.max_calls_per_minute:
            wait_time = timestamps[0] - cutoff + self.config.backoff_seconds
            return False, f"Rate limit exceeded. Wait {wait_time:.1f}s before next call."
        
        # Check per-tool limit
        if tool_name:
            tool_count = self._get_tool_count(conversation_id, tool_name)
            if tool_count >= self.config.max_calls_per_tool:
                return False, f"Tool '{tool_name}' limit exceeded ({self.config.max_calls_per_tool} calls)."
        
        return True, "OK"
    
    def record_call(self, conversation_id: str, tool_name: str | None = None) -> None:
        """Record a successful call."""
        timestamps = self._get_timestamps(conversation_id)
        timestamps.append(time.time())
        
        if tool_name:
            self._increment_tool_count(conversation_id, tool_name)
    
    def reset(self, conversation_id: str) -> None:
        """Reset limits for a conversation."""
        if conversation_id in self._call_timestamps:
            del self._call_timestamps[conversation_id]
        if conversation_id in self._tool_counts:
            del self._tool_counts[conversation_id]


class RateLimitMiddleware:
    """
    Rate limit middleware for agents.
    
    Usage:
        middleware = RateLimitMiddleware()
        
        # Before each tool call:
        allowed, msg = middleware.check(conversation_id, tool_name)
        if not allowed:
            return msg
    """
    
    def __init__(self, config: RateLimitConfig | None = None):
        self.limiter = RateLimiter(config)
    
    def check(self, conversation_id: str, tool_name: str | None = None) -> tuple[bool, str]:
        """Check if call is allowed."""
        return self.limiter.check_rate_limit(conversation_id, tool_name)
    
    def record(self, conversation_id: str, tool_name: str | None = None) -> None:
        """Record a successful call."""
        self.limiter.record_call(conversation_id, tool_name)
    
    def reset(self, conversation_id: str) -> None:
        """Reset limits."""
        self.limiter.reset(conversation_id)

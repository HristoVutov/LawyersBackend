"""
Max Iterations Guard - Prevents infinite loops in agent execution.

Provides:
- Per-conversation iteration tracking
- Configurable max iterations
- Automatic force-exit on limit
"""
from dataclasses import dataclass


@dataclass
class MaxIterationsConfig:
    """Configuration for max iterations guard."""
    max_iterations: int = 25
    warning_threshold: int = 20


class MaxIterationsGuard:
    """
    Guard against infinite loops in agent execution.
    """
    
    def __init__(self, config: MaxIterationsConfig | None = None):
        self.config = config or MaxIterationsConfig()
        self._iteration_counts: dict[str, int] = {}
    
    def get_iteration(self, conversation_id: str) -> int:
        """Get current iteration count."""
        return self._iteration_counts.get(conversation_id, 0)
    
    def increment(self, conversation_id: str) -> int:
        """Increment and return new count."""
        count = self.get_iteration(conversation_id) + 1
        self._iteration_counts[conversation_id] = count
        return count
    
    def check(self, conversation_id: str) -> tuple[bool, str | None]:
        """
        Check if iteration is allowed.
        
        Returns:
            (allowed, warning_message) - warning_message is non-None near limit
        """
        count = self.get_iteration(conversation_id)
        
        if count >= self.config.max_iterations:
            return False, f"⚠️ Max iterations ({self.config.max_iterations}) reached. Forcing exit."
        
        if count >= self.config.warning_threshold:
            remaining = self.config.max_iterations - count
            return True, f"⚠️ Approaching iteration limit. {remaining} iterations remaining."
        
        return True, None
    
    def reset(self, conversation_id: str) -> None:
        """Reset iteration count for a conversation."""
        if conversation_id in self._iteration_counts:
            del self._iteration_counts[conversation_id]


class MaxIterationsMiddleware:
    """
    Max iterations middleware for agents.
    
    Integrates with LangGraph agent loop to prevent infinite execution.
    
    Usage:
        middleware = MaxIterationsMiddleware()
        
        # At start of each iteration:
        count = middleware.increment(conversation_id)
        allowed, msg = middleware.check(conversation_id)
        if not allowed:
            force_exit()
    """
    
    def __init__(self, max_iterations: int = 25, warning_threshold: int = 20):
        self.guard = MaxIterationsGuard(MaxIterationsConfig(
            max_iterations=max_iterations,
            warning_threshold=warning_threshold
        ))
    
    def increment(self, conversation_id: str) -> int:
        """Increment iteration count."""
        return self.guard.increment(conversation_id)
    
    def check(self, conversation_id: str) -> tuple[bool, str | None]:
        """Check if more iterations allowed."""
        return self.guard.check(conversation_id)
    
    def reset(self, conversation_id: str) -> None:
        """Reset for new conversation/task."""
        self.guard.reset(conversation_id)
    
    def get_count(self, conversation_id: str) -> int:
        """Get current count."""
        return self.guard.get_iteration(conversation_id)

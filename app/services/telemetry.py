"""
Telemetry Service - Anonymous usage analytics for Lawyers Dashboard.

Sends non-PII usage events to the cloud analytics service.
Events are batched and sent asynchronously (fire-and-forget).
"""
import asyncio
import hashlib
import uuid
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict
import httpx

from app.config import get_settings


@dataclass
class TelemetryEvent:
    """A single telemetry event."""
    event_type: str  # chat_message, agent_run, tool_call, error
    user_id: str     # Anonymous hash
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    metadata: Optional[dict] = None
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TokenUsageRecord:
    """Token usage from a single LLM call."""
    thread_id: str
    agent_name: str
    model_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_cents: Optional[float] = None
    metadata: Optional[dict] = None
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


# Token cost per 1M tokens (approximate costs for Gemini models)
# Token cost per 1M tokens (approximate costs for Gemini models)
TOKEN_COSTS = {
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},  # $ per 1M tokens
    "default": {"input": 0.075, "output": 0.30},
}

def load_pricing_config():
    """Load pricing from JSON file or return default."""
    import json
    from pathlib import Path
    
    # Try multiple locations for pricing.json
    possible_paths = [
        Path(__file__).parent.parent.parent / "analytics" / "pricing.json", # Dev env
        Path("pricing.json"), # Root
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                with open(path, "r") as f:
                    pricing = json.load(f)
                    # Convert to internal format {"input": X, "output": Y}
                    normalized = {}
                    for model, costs in pricing.items():
                        normalized[model] = {
                            "input": costs.get("input_cost_per_m", 0.0),
                            "output": costs.get("output_cost_per_m", 0.0)
                        }
                    return normalized
            except Exception:
                pass
                
    return TOKEN_COSTS

# Load once at startup
CURRENT_PRICING = load_pricing_config()

def calculate_cost_cents(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in cents based on model and token counts."""
    costs = CURRENT_PRICING.get(model_name, CURRENT_PRICING.get("default"))
    if not costs:
        return 0.0
    
    # Cost = (tokens / 1M) * cost_per_1M * 100 (to cents)
    input_cost = (input_tokens / 1_000_000) * costs["input"] * 100
    output_cost = (output_tokens / 1_000_000) * costs["output"] * 100
    return round(input_cost + output_cost, 4)


class TelemetryService:
    """
    Async telemetry service with batching and graceful failure.
    
    Usage:
        telemetry = get_telemetry()
        await telemetry.track("chat_message", metadata={"agent": "orchestrator"})
        await telemetry.track_token_usage(thread_id, agent, model, input, output)
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: str = "",
        enabled: bool = True,
        batch_size: int = 10,
        flush_interval_seconds: float = 30.0,
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.enabled = enabled
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        
        self._event_queue: list[TelemetryEvent] = []
        self._token_queue: list[TokenUsageRecord] = []
        self._user_id: str = self._generate_anonymous_id()
        self._flush_task: Optional[asyncio.Task] = None
        self._client: Optional[httpx.AsyncClient] = None
    
    def _generate_anonymous_id(self) -> str:
        """Generate a stable anonymous user ID (not PII)."""
        try:
            import platform
            machine_id = f"{platform.node()}-{uuid.getnode()}"
        except Exception:
            machine_id = str(uuid.uuid4())
        
        return hashlib.sha256(machine_id.encode()).hexdigest()[:32]
    
    @property
    def user_id(self) -> str:
        return self._user_id
    
    async def track(
        self,
        event_type: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Queue an event for sending to analytics."""
        if not self.enabled:
            return
        
        event = TelemetryEvent(
            event_type=event_type,
            user_id=self._user_id,
            metadata=metadata,
        )
        self._event_queue.append(event)
        
        if len(self._event_queue) >= self.batch_size:
            asyncio.create_task(self.flush_events())
    
    async def track_token_usage(
        self,
        thread_id: str,
        agent_name: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Track token usage from an LLM call.
        
        Automatically calculates cost based on model pricing.
        """
        if not self.enabled:
            return
        
        total_tokens = input_tokens + output_tokens
        cost_cents = calculate_cost_cents(model_name, input_tokens, output_tokens)
        
        record = TokenUsageRecord(
            thread_id=thread_id,
            agent_name=agent_name,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_cents=cost_cents,
            metadata=metadata,
        )
        self._token_queue.append(record)
        
        if len(self._token_queue) >= self.batch_size:
            asyncio.create_task(self.flush_tokens())
    
    async def flush_events(self) -> None:
        """Send queued events to the analytics endpoint."""
        if not self._event_queue or not self.enabled:
            return
        
        events_to_send = self._event_queue.copy()
        self._event_queue.clear()
        
        try:
            await self._ensure_client()
            headers = self._get_headers()
            
            await self._client.post(
                f"{self.endpoint}/api/events",
                json={"events": [e.to_dict() for e in events_to_send]},
                headers=headers,
            )
        except Exception:
            pass  # Fire-and-forget
    
    async def flush_tokens(self) -> None:
        """Send queued token usage to the analytics endpoint."""
        if not self._token_queue or not self.enabled:
            return
        
        records_to_send = self._token_queue.copy()
        self._token_queue.clear()
        
        try:
            await self._ensure_client()
            headers = self._get_headers()
            
            await self._client.post(
                f"{self.endpoint}/api/usage/tokens",
                json={"records": [r.to_dict() for r in records_to_send]},
                headers=headers,
            )
        except Exception:
            pass  # Fire-and-forget
    
    async def flush(self) -> None:
        """Flush both events and token usage."""
        await self.flush_events()
        await self.flush_tokens()
    
    async def _ensure_client(self) -> None:
        """Ensure HTTP client exists."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
    
    def _get_headers(self) -> dict:
        """Get headers for API requests."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def start_background_flush(self) -> None:
        """Start background task that flushes periodically."""
        if self._flush_task is not None:
            return
        
        async def _flush_loop():
            while True:
                await asyncio.sleep(self.flush_interval)
                await self.flush()
        
        self._flush_task = asyncio.create_task(_flush_loop())
    
    async def shutdown(self) -> None:
        """Flush remaining events and close connections."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        await self.flush()
        
        if self._client:
            await self._client.aclose()


# Global instance
_telemetry: Optional[TelemetryService] = None


def get_telemetry() -> TelemetryService:
    """Get the global telemetry service instance."""
    global _telemetry
    if _telemetry is None:
        settings = get_settings()
        _telemetry = TelemetryService(
            endpoint=getattr(settings, 'telemetry_endpoint', 'http://localhost:8001'),
            api_key=getattr(settings, 'telemetry_api_key', ''),
            enabled=getattr(settings, 'telemetry_enabled', False),
        )
    return _telemetry


async def track_event(event_type: str, metadata: Optional[dict] = None) -> None:
    """Convenience function to track an event."""
    await get_telemetry().track(event_type, metadata)


async def track_token_usage(
    thread_id: str,
    agent_name: str,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    metadata: Optional[dict] = None,
) -> None:
    """Convenience function to track token usage."""
    await get_telemetry().track_token_usage(
        thread_id, agent_name, model_name, input_tokens, output_tokens, metadata
    )


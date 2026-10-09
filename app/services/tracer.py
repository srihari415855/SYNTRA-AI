"""Execution Tracer: Real-Time Event Timeline & Telemetry Logger.

Captures actual execution events with high-precision millisecond timestamps,
latency tracking, model IDs, status, and safe diagnostics for both normal executions
and model migrations.
"""

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    timestamp_ms: float
    elapsed_ms: float
    event_name: str
    status: str  # OK, INFO, WARNING, ERROR, SUCCESS
    description: str
    component: str  # Router, Catalogue, Provider, TrustLayer, ContextDNA, MigrationEngine
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionTrace(BaseModel):
    session_id: str
    user_id: str
    start_time: float
    total_elapsed_ms: float = 0.0
    events: List[TraceEvent] = Field(default_factory=list)


class ExecutionTracer:
    """Manages recording of structured execution trace timelines."""

    def __init__(self, session_id: str = "sess_default", user_id: str = "guest_pilot"):
        self.session_id = session_id
        self.user_id = user_id
        self.start_time = time.perf_counter()
        self.events: List[TraceEvent] = []

    def log(
        self,
        event_name: str,
        description: str,
        component: str,
        status: str = "OK",
        metadata: Optional[Dict[str, Any]] = None
    ) -> TraceEvent:
        """Appends a timestamped event with calculated elapsed time."""
        now = time.perf_counter()
        elapsed = round((now - self.start_time) * 1000, 2)
        event = TraceEvent(
            timestamp_ms=round(now * 1000, 2),
            elapsed_ms=elapsed,
            event_name=event_name,
            status=status,
            description=description,
            component=component,
            metadata=metadata or {}
        )
        self.events.append(event)
        return event

    def to_dict(self) -> Dict[str, Any]:
        """Serializes trace for API responses."""
        total_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "total_elapsed_ms": total_ms,
            "event_count": len(self.events),
            "events": [e.model_dump() for e in self.events]
        }

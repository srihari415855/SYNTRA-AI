"""Agent State Schema."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """
    Contract 3: Canonical, mutable representation of system knowledge during a conversation turn.
    Updated sequentially by every action taken (user input -> model thought -> tool result).
    """
    session_id: str = Field(..., description="Unique identifier for this specific multi-step session.")
    current_context: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Filtered, actionable list of messages including tool results and model outputs."
    )
    last_tool_call: Optional[Tuple[str, Dict[str, Any]]] = Field(
        default=None,
        description="Name and arguments of the most recent function call executed in this turn."
    )
    final_status: Literal["IN_PROGRESS", "COMPLETE", "FAILED"] = Field(
        default="IN_PROGRESS",
        description="Indicates if the system is still executing steps or ready to render a final answer."
    )
    iteration_count: int = Field(default=0, description="Loop iteration counter enforcing MaxDepth.")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def update_timestamp(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

"""System Responses and Validation Report Schemas."""

from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field
from app.schemas.tool_call import ToolCall, ToolResult


class ValidationReport(TypedDict):
    """Contract for validation feedback matching Principal Engineer ABC review."""
    is_valid: bool
    errors: List[str]
    suggestions: Optional[List[str]]


class RouteDecision(BaseModel):
    """Metadata regarding how the request was routed."""
    route_type: str = Field(..., description="'deterministic' or 'model_classified'")
    category: str = Field(..., description="'general', 'coding', 'reasoning', 'complex_reasoning', or 'direct_tool'")
    target_model: Optional[str] = Field(default=None, description="Model selected for execution")
    rationale: str = Field(default="", description="Explanation or rule matched")


class ModelResponse(BaseModel):
    """Normalized response returned by the Model Interface Layer."""
    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: Optional[str] = "stop"
    raw_response: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """End-user response payload for the API Gateway."""
    session_id: str
    user_id: str
    final_output: str
    status: str
    route: RouteDecision
    iterations: int
    executed_tools: List[ToolResult] = Field(default_factory=list)
    execution_time_ms: float = 0.0

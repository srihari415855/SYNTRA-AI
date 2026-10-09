"""Tool Definition and Tool Call Schemas."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    """
    Contract 2: Defines an external capability, providing the LLM with function-calling metadata.
    """
    name: str = Field(..., description="Unique identifier for the tool, e.g. 'calculator'.")
    description: str = Field(..., description="Explanation of what the tool does for LLM reasoning.")
    parameters: Dict[str, Any] = Field(
        ...,
        description="JSON schema defining acceptable arguments and their types."
    )


class ToolCall(BaseModel):
    """Structured representation of a function call requested by the LLM."""
    id: str = Field(default="", description="Unique identifier of this call invocation.")
    name: str = Field(..., description="Name of the tool to be executed.")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Parsed arguments matching tool parameters.")


class ToolResult(BaseModel):
    """Structured result returned by the Tool Runtime Executor."""
    tool_name: str
    call_id: Optional[str] = None
    output: Any
    is_error: bool = False
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0

"""API Request and Response Models for Dynamic Model-Routing Platform."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, EmailStr, Field
from app.schemas.responses import ChatResponse


# =============================================================================
# Authentication Schemas
# =============================================================================

class RegisterRequest(BaseModel):
    """User registration payload."""
    email: str = Field(..., description="Valid user email address.")
    name: str = Field(..., min_length=2, description="Display name or full name.")
    password: str = Field(..., min_length=8, description="Password meeting complexity criteria.")


class LoginRequest(BaseModel):
    """User login payload."""
    email: str = Field(..., description="Registered email.")
    password: str = Field(..., description="User password.")


class PasswordResetRequest(BaseModel):
    """Password reset request payload."""
    email: str = Field(...)
    new_password: str = Field(..., min_length=8)


class UserPreferencesRequest(BaseModel):
    """User settings and ranking preferences."""
    ranking_mode: Optional[str] = "best_value"
    prefer_free: Optional[bool] = True
    prefer_local: Optional[bool] = True


class AuthResponse(BaseModel):
    """Authentication token and profile response."""
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    name: str
    is_demo: bool = False
    preferences: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Request Understanding & Recommendation Schemas
# =============================================================================

class UnderstandRequest(BaseModel):
    """Request payload for dynamic understanding pipeline."""
    prompt: str = Field(..., min_length=1)
    has_image: bool = False
    has_audio: bool = False


class RecommendRequest(BaseModel):
    """Request payload for dynamic model recommendation ranking."""
    prompt: str = Field(..., min_length=1)
    ranking_mode: str = Field(default="best_value")  # free, paid, lowest_price, best_performance, best_value
    has_image: bool = False
    has_audio: bool = False
    max_budget_cents: Optional[float] = None


class ExecuteRequest(BaseModel):
    """Request payload for executing a model turn."""
    prompt: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    selected_model: Optional[str] = None
    ranking_mode: str = "best_value"
    image_base64: Optional[str] = None
    has_audio: bool = False


# =============================================================================
# General & Legacy Schemas
# =============================================================================

class ChatRequest(BaseModel):
    """Legacy/direct payload for POST /api/v1/chat endpoint."""
    user_id: str = Field(...)
    session_id: Optional[str] = None
    prompt: str = Field(..., min_length=1)
    history: List[Dict[str, Any]] = Field(default_factory=list)


class ToolSummary(BaseModel):
    """Summary of registered tool."""
    name: str
    description: str
    parameters: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check status."""
    status: str
    brand: str
    environment: str
    database: str
    ollama_endpoint: str
    registered_tools: int
    available_models: int


# =============================================================================
# Jury Demo, Trust Layer Verification & Quality Recovery Schemas
# =============================================================================

class DemoSimulateFailureRequest(BaseModel):
    """Payload for Jury Demonstration Mode with controlled failure injection."""
    prompt: str = Field(..., min_length=1)
    source_model: str = Field(default="gpt-4o")
    simulated_failure: str = Field(
        default="Rate limit exceeded: 429 Too Many Requests on primary provider",
        description="Controlled failure message to simulate and classify."
    )
    prefer_free_only: bool = Field(default=True)
    verification_policy: str = Field(default="standard")


class VerifyRequest(BaseModel):
    """Direct verification of prompt and output text."""
    prompt: str = Field(..., min_length=1)
    output: str = Field(..., min_length=1)
    task_category: str = Field(default="general")
    policy: str = Field(default="standard")


class FeedbackRequest(BaseModel):
    """User post-response feedback and recovery action."""
    session_id: str = Field(...)
    prompt: str = Field(...)
    current_output: str = Field(...)
    feedback_type: str = Field(
        ...,
        description="One of: retry, improve, verify, switch_model, report_issue"
    )
    notes: Optional[str] = None
    target_model: Optional[str] = None


class ReviewSubmissionRequest(BaseModel):
    """Payload for submitting 1-5 star review and multidimensional feedback."""
    session_id: str = Field(...)
    model_id: str = Field(...)
    provider: str = Field(...)
    task_category: str = Field(default="general")
    rating: int = Field(..., ge=1, le=5)
    is_correct: Optional[bool] = None
    is_useful: Optional[bool] = None
    is_complete: Optional[bool] = None
    is_fast: Optional[bool] = None
    cost_acceptable: Optional[bool] = None
    verification_status: Optional[str] = None
    feedback_text: Optional[str] = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0


class DashboardStatsResponse(BaseModel):
    """Real system-wide aggregated telemetry and performance metrics."""
    total_requests: int
    successful_executions: int
    failed_executions: int
    success_rate_pct: float
    model_usage_distribution: Dict[str, int]
    category_distribution: Dict[str, int]
    free_vs_paid: Dict[str, int]
    total_tokens_consumed: int
    estimated_tokens_saved: int
    total_cost_usd: float
    latency: Dict[str, float]
    reviews_count: int
    average_rating: float


class ApiKeysUpdateRequest(BaseModel):
    """Payload for updating external model provider API keys."""
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None



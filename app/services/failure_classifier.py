"""Adaptive Provider Failure Classifier and Recovery Action Resolver."""

import re
from enum import Enum
from typing import Any, Dict, Optional
import httpx
from pydantic import BaseModel

from app.core.exceptions import (
    AISystemBaseException,
    ModelInferenceError,
    ModelRateLimitError,
    ModelUnavailableError,
    ToolArgumentValidationError,
    ToolExecutionError,
    ToolNotFoundError,
)


class FailureCategory(str, Enum):
    """Canonical provider and execution failure categories."""
    TRANSIENT_NETWORK_FAILURE = "TRANSIENT_NETWORK_FAILURE"
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    AUTHENTICATION_FAILURE = "AUTHENTICATION_FAILURE"
    BILLING_RESTRICTION = "BILLING_RESTRICTION"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    INVALID_REQUEST = "INVALID_REQUEST"
    CONTEXT_LIMIT_EXCEEDED = "CONTEXT_LIMIT_EXCEEDED"
    OUTPUT_VALIDATION_FAILURE = "OUTPUT_VALIDATION_FAILURE"
    TOOL_EXECUTION_FAILURE = "TOOL_EXECUTION_FAILURE"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"


class FailureAnalysis(BaseModel):
    """Structured failure diagnosis and recommended recovery strategy."""
    category: FailureCategory
    error_message: str
    status_code: Optional[int] = None
    is_retryable: bool
    requires_fallback: bool
    recovery_action: str
    safe_user_message: str
    details: Dict[str, Any] = {}


class FailureClassifier:
    """Classifies runtime exceptions and provider responses into actionable failure categories."""

    def classify(
        self,
        exc: Exception,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> FailureAnalysis:
        """Analyzes exception types, HTTP status codes, and error payloads."""
        err_str = str(exc).lower()

        # 1. Rate Limiting (429)
        if isinstance(exc, ModelRateLimitError) or "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
            return FailureAnalysis(
                category=FailureCategory.RATE_LIMITED,
                error_message=str(exc),
                status_code=429,
                is_retryable=True,
                requires_fallback=False,
                recovery_action="Apply exponential backoff with jitter or await Retry-After window.",
                safe_user_message=f"Provider {provider_name or 'engine'} is temporarily rate-limited. Backoff retry in progress."
            )

        # 2. Quota Exhaustion
        quota_keywords = ["quota", "insufficient_quota", "credit balance", "exceeded your current quota", "billing limit reached"]
        if any(k in err_str for k in quota_keywords):
            return FailureAnalysis(
                category=FailureCategory.QUOTA_EXHAUSTED,
                error_message=str(exc),
                status_code=402,
                is_retryable=False,
                requires_fallback=True,
                recovery_action="Initiate immediate Context DNA model migration to an eligible alternative.",
                safe_user_message=f"Usage quota on {provider_name or 'model'} exhausted. Migrating work to backup engine."
            )

        # 3. Authentication & Billing Restrictions
        if "401" in err_str or "unauthorized" in err_str or "invalid api key" in err_str or "incorrect api key" in err_str:
            return FailureAnalysis(
                category=FailureCategory.AUTHENTICATION_FAILURE,
                error_message=str(exc),
                status_code=401,
                is_retryable=False,
                requires_fallback=True,
                recovery_action="Surface credential configuration error and failover to unauthenticated local model.",
                safe_user_message="API authentication key missing or invalid. Rerouting to verified local runtime."
            )

        if "403" in err_str or "forbidden" in err_str or "billing" in err_str:
            return FailureAnalysis(
                category=FailureCategory.BILLING_RESTRICTION,
                error_message=str(exc),
                status_code=403,
                is_retryable=False,
                requires_fallback=True,
                recovery_action="Enforce cost policy and fallback to compatible free/local engine.",
                safe_user_message="Provider account has billing restrictions. Transferring task to eligible engine."
            )

        # 4. Context Window Overflow
        context_keywords = ["context length", "maximum context length", "too many tokens", "context_length_exceeded", "prompt is too long"]
        if any(k in err_str for k in context_keywords):
            return FailureAnalysis(
                category=FailureCategory.CONTEXT_LIMIT_EXCEEDED,
                error_message=str(exc),
                status_code=400,
                is_retryable=False,
                requires_fallback=True,
                recovery_action="Rebuild compressed Context DNA packet with aggressive token reduction or select high-context model.",
                safe_user_message="Conversation exceeded model context window. Compressing context for larger-window model."
            )

        # 5. Unsupported Capabilities
        if "unsupported modality" in err_str or "vision not supported" in err_str or "does not support tools" in err_str:
            return FailureAnalysis(
                category=FailureCategory.UNSUPPORTED_CAPABILITY,
                error_message=str(exc),
                status_code=400,
                is_retryable=False,
                requires_fallback=True,
                recovery_action="Filter catalogue for models matching required task modality and re-dispatch.",
                safe_user_message="Requested engine does not support this task's modalities. Migrating to compatible engine."
            )

        # 6. Tool Execution Failures
        if isinstance(exc, (ToolExecutionError, ToolArgumentValidationError, ToolNotFoundError)):
            return FailureAnalysis(
                category=FailureCategory.TOOL_EXECUTION_FAILURE,
                error_message=str(exc),
                is_retryable=False,
                requires_fallback=False,
                recovery_action="Capture tool error feedback and inject into context for model self-healing.",
                safe_user_message="Tool invocation error captured. Model instructed to self-correct."
            )

        # 7. Transient Network Failures
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError)):
            return FailureAnalysis(
                category=FailureCategory.TRANSIENT_NETWORK_FAILURE,
                error_message=str(exc),
                is_retryable=True,
                requires_fallback=False,
                recovery_action="Retry connection via Tenacity exponential backoff policy.",
                safe_user_message="Temporary network hiccup connecting to provider. Retrying request."
            )

        # 8. Provider Unavailable / Service Down
        if isinstance(exc, ModelUnavailableError) or "502" in err_str or "503" in err_str or "504" in err_str:
            return FailureAnalysis(
                category=FailureCategory.PROVIDER_UNAVAILABLE,
                error_message=str(exc),
                status_code=503,
                is_retryable=True,
                requires_fallback=True,
                recovery_action="Probe fallback status and transfer task if provider remains unresponsive.",
                safe_user_message=f"Provider {provider_name or 'endpoint'} is temporarily unavailable."
            )

        # 9. Generic / Unknown
        return FailureAnalysis(
            category=FailureCategory.UNKNOWN_FAILURE,
            error_message=str(exc),
            is_retryable=False,
            requires_fallback=False,
            recovery_action="Log diagnostic trace and return safe controlled error message.",
            safe_user_message="An unexpected execution error occurred. Diagnostics recorded."
        )


failure_classifier = FailureClassifier()

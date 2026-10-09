"""Tenacity Retry Policies and Strategies for Resilient External Operations."""

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)
from app.core.config import settings
from app.core.exceptions import ModelRateLimitError, ModelUnavailableError
from app.core.logging import get_logger

logger = get_logger("ai_system.retry")


def get_model_retry_policy() -> AsyncRetrying:
    """
    Constructs an asynchronous Tenacity retry policy with exponential backoff and jitter.
    Catches transient HTTP network errors, rate limits (429), and temporary server errors.
    """
    return AsyncRetrying(
        stop=stop_after_attempt(settings.RETRY_MAX_ATTEMPTS),
        wait=wait_random_exponential(
            multiplier=settings.RETRY_MIN_WAIT_SECONDS,
            max=settings.RETRY_MAX_WAIT_SECONDS
        ),
        retry=retry_if_exception_type((
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            ModelRateLimitError,
            ModelUnavailableError,
        )),
        before_sleep=before_sleep_log(logger, log_level=20),  # INFO level
        reraise=True,
    )

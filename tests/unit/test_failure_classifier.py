"""Unit tests for Adaptive Provider Failure Classifier."""

import unittest
import httpx
from app.core.exceptions import ModelRateLimitError, ModelUnavailableError
from app.services.failure_classifier import (
    FailureCategory,
    failure_classifier,
)


class TestFailureClassifier(unittest.TestCase):
    """Validates classification of exceptions and status codes across failure categories."""

    def test_classify_rate_limit(self):
        exc = ModelRateLimitError("429 Too Many Requests: Rate limit exceeded", model_name="claude-3-5-sonnet")
        analysis = failure_classifier.classify(exc, provider_name="Anthropic")
        self.assertEqual(analysis.category, FailureCategory.RATE_LIMITED)
        self.assertTrue(analysis.is_retryable)
        self.assertFalse(analysis.requires_fallback)

    def test_classify_quota_exhaustion(self):
        exc = Exception("Error 402: You have exceeded your current quota / billing limit reached")
        analysis = failure_classifier.classify(exc, provider_name="OpenAI")
        self.assertEqual(analysis.category, FailureCategory.QUOTA_EXHAUSTED)
        self.assertFalse(analysis.is_retryable)
        self.assertTrue(analysis.requires_fallback)

    def test_classify_auth_failure(self):
        exc = Exception("HTTP 401 Unauthorized: Invalid API Key provided")
        analysis = failure_classifier.classify(exc, provider_name="OpenAI")
        self.assertEqual(analysis.category, FailureCategory.AUTHENTICATION_FAILURE)
        self.assertFalse(analysis.is_retryable)
        self.assertTrue(analysis.requires_fallback)

    def test_classify_context_limit(self):
        exc = Exception("Maximum context length exceeded: prompt is too long for 8k window")
        analysis = failure_classifier.classify(exc, provider_name="Meta")
        self.assertEqual(analysis.category, FailureCategory.CONTEXT_LIMIT_EXCEEDED)
        self.assertFalse(analysis.is_retryable)
        self.assertTrue(analysis.requires_fallback)

    def test_classify_transient_network(self):
        exc = httpx.ConnectTimeout("Connection timed out after 10000ms")
        analysis = failure_classifier.classify(exc)
        self.assertEqual(analysis.category, FailureCategory.TRANSIENT_NETWORK_FAILURE)
        self.assertTrue(analysis.is_retryable)


if __name__ == "__main__":
    unittest.main()

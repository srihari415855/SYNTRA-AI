"""Unit tests for Trust Layer Multi-Stage Output Verification Engine."""

import unittest
from app.services.verification import (
    FindingSeverity,
    VerificationDecision,
    VerificationPolicy,
    verification_engine,
)


class TestTrustLayerVerification(unittest.TestCase):
    """Validates Stages A to E of Trust Layer output verification."""

    def test_stage_a_rejects_empty_output(self):
        """Stage A must reject empty or blank responses."""
        report = verification_engine.verify(
            prompt="Write a function",
            output="   ",
            task_category="coding"
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.decision, VerificationDecision.REJECTED)
        self.assertFalse(report.stage_a_integrity)

    def test_stage_a_detects_provider_raw_errors(self):
        """Stage A must detect leaked provider error JSON or status strings."""
        error_output = '{"error": {"message": "Rate limit exceeded for organization", "type": "tokens"}}'
        report = verification_engine.verify(
            prompt="Summarize article",
            output=error_output,
            task_category="summarization"
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.decision, VerificationDecision.REJECTED)
        self.assertFalse(report.stage_a_integrity)

    def test_stage_b_negative_constraint_enforcement(self):
        """Stage B must detect violations of negative instructions."""
        prompt = "Explain quantum computing without mathematics"
        violating_output = "Quantum computing relies heavily on complex mathematics and linear algebra."
        report = verification_engine.verify(
            prompt=prompt,
            output=violating_output,
            task_category="general",
            policy=VerificationPolicy.STRICT
        )
        self.assertFalse(report.stage_b_compliance)

    def test_stage_c_python_ast_syntax_validation(self):
        """Stage C parses fenced Python blocks via ast.parse and detects syntax errors."""
        valid_code = (
            "Here is the solution:\n\n"
            "```python\n"
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
            "```\n"
        )
        valid_report = verification_engine.verify(
            prompt="Write an add function",
            output=valid_code,
            task_category="coding"
        )
        self.assertTrue(valid_report.stage_c_task_validation)
        self.assertEqual(valid_report.decision, VerificationDecision.VERIFIED)

        # Broken syntax test
        broken_code = (
            "Here is the broken code:\n\n"
            "```python\n"
            "def broken_function(\n"
            "    return a +\n"
            "```\n"
        )
        broken_report = verification_engine.verify(
            prompt="Write code",
            output=broken_code,
            task_category="coding"
        )
        self.assertFalse(broken_report.stage_c_task_validation)
        self.assertEqual(broken_report.decision, VerificationDecision.REJECTED)

    def test_stage_c_deterministic_math_check(self):
        """Stage C verifies claimed arithmetic results against actual deterministic calculation."""
        correct_output = "The calculation is 25 * 25 = 625."
        correct_report = verification_engine.verify(
            prompt="Calculate 25 * 25",
            output=correct_output,
            task_category="math"
        )
        self.assertTrue(correct_report.stage_c_task_validation)

        # Math hallucination detection
        incorrect_output = "The calculation is 25 * 25 = 650."
        incorrect_report = verification_engine.verify(
            prompt="Calculate 25 * 25",
            output=incorrect_output,
            task_category="math"
        )
        self.assertFalse(incorrect_report.stage_c_task_validation)
        self.assertEqual(incorrect_report.decision, VerificationDecision.REJECTED)

    def test_stage_e_policy_matrix(self):
        """Low latency policy only checks Stage A, whereas Strict policy enforces all checks."""
        output = "Simple answer with minor note."
        low_latency = verification_engine.verify(
            prompt="test",
            output=output,
            policy=VerificationPolicy.LOW_LATENCY
        )
        self.assertTrue(low_latency.passed)


if __name__ == "__main__":
    unittest.main()

"""Unit Tests for Intent Router."""

import unittest
from app.orchestrator.router import IntentRouter


class TestIntentRouter(unittest.TestCase):
    """Verifies deterministic fast matching and model-based intent categories."""

    def setUp(self):
        self.router = IntentRouter()

    def test_deterministic_calculator_pattern(self):
        decision = self.router.route("calculate 45 * 2")
        self.assertEqual(decision.route_type, "deterministic")
        self.assertEqual(decision.category, "direct_tool")
        self.assertEqual(decision.target_model, "calculator")

    def test_deterministic_system_info_pattern(self):
        decision = self.router.route("system info")
        self.assertEqual(decision.route_type, "deterministic")
        self.assertEqual(decision.category, "direct_tool")
        self.assertEqual(decision.target_model, "system_info")

    def test_coding_intent_classification(self):
        decision = self.router.route("Write a python function to parse json")
        self.assertEqual(decision.route_type, "model_classified")
        self.assertEqual(decision.category, "coding")

    def test_complex_reasoning_intent(self):
        decision = self.router.route("Please provide a mathematical proof for Fermat's theorem")
        self.assertEqual(decision.route_type, "model_classified")
        self.assertEqual(decision.category, "complex_reasoning")

    def test_general_intent_fallback(self):
        decision = self.router.route("Hello, can you help me plan my day?")
        self.assertEqual(decision.route_type, "model_classified")
        self.assertEqual(decision.category, "general")


if __name__ == "__main__":
    unittest.main()

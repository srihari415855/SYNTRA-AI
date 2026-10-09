"""Integration tests for Seamless Model Migration and Jury Demonstration Mode."""

import unittest
from fastapi.testclient import TestClient
from app.main import app
from app.services.migration import model_migration_engine


class TestMigrationAndDemo(unittest.TestCase):
    """Integration test suite for failover migration, Jury Demo, and verification APIs."""

    def setUp(self):
        self.client = TestClient(app)

    def test_direct_model_migration_engine(self):
        """Validates execution of the 12-step failover workflow in model_migration_engine."""
        import asyncio
        async def run_migration():
            return await model_migration_engine.execute_failover_migration(
                session_id="test_integ_migration",
                user_id="test_pilot",
                prompt="Write a Python function to compute Fibonacci sequence",
                source_model="gpt-4o",
                trigger_reason="SIMULATED_QUOTA_EXHAUSTED",
                simulated_failure="Quota exceeded on OpenAI billing account",
                prefer_free_only=True
            )

        result = asyncio.run(run_migration())
        self.assertTrue(result.success)
        self.assertEqual(result.source_model, "gpt-4o")
        self.assertNotEqual(result.target_model, "gpt-4o")
        self.assertNotEqual(result.target_provider, "OpenAI")  # Circuit breaker worked!
        self.assertTrue(result.circuit_breaker_active)
        self.assertIn("def process_pipeline", result.continuation_output)
        self.assertTrue(result.verification_report["passed"])
        self.assertGreater(len(result.execution_trace["events"]), 5)

    def test_jury_demo_api_endpoint(self):
        """Validates POST /api/v1/demo/simulate-failure-migration end-to-end API response."""
        payload = {
            "prompt": "Calculate 15 * 15 and explain the steps",
            "source_model": "claude-3-5-sonnet",
            "simulated_failure": "Rate limit exceeded (429) on Anthropic endpoint",
            "prefer_free_only": True
        }
        res = self.client.post("/api/v1/demo/simulate-failure-migration", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertTrue(data["success"])
        self.assertEqual(data["source_model"], "claude-3-5-sonnet")
        self.assertNotEqual(data["target_provider"], "Anthropic")  # Circuit breaker rerouted
        self.assertIn("context_dna", data)
        self.assertIn("handover_packet", data)
        self.assertIn("execution_trace", data)
        self.assertIn("verification_report", data)
        self.assertTrue(data["verification_report"]["passed"])

    def test_direct_verify_api_endpoint(self):
        """Validates POST /api/v1/verify returns structured Trust Layer report."""
        payload = {
            "prompt": "Write Python code",
            "output": "```python\ndef test():\n    return True\n```",
            "task_category": "coding",
            "policy": "strict"
        }
        res = self.client.post("/api/v1/verify", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["decision"], "VERIFIED")
        self.assertTrue(data["passed"])
        self.assertTrue(data["stage_a_integrity"])
        self.assertTrue(data["stage_c_task_validation"])

    def test_feedback_recovery_api_endpoint(self):
        """Validates POST /api/v1/feedback handles user quality recovery triggers."""
        payload = {
            "session_id": "test_sess_feedback",
            "prompt": "Build an API endpoint",
            "current_output": "Sample output text",
            "feedback_type": "verify",
            "notes": "Verify this output meets requirements"
        }
        res = self.client.post("/api/v1/feedback", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["feedback_type"], "verify")
        self.assertIn("verification_report", data)


if __name__ == "__main__":
    unittest.main()

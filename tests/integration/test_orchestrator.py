"""Integration Tests for the Orchestration Engine."""

import unittest
from app.orchestrator.engine import OrchestratorEngine
from app.schemas.context import PromptContext
from app.storage.repository import state_repository


class TestOrchestratorEngine(unittest.IsolatedAsyncioTestCase):
    """Tests the full multi-turn execution loop, state persistence, and guardrails."""

    def setUp(self):
        self.engine = OrchestratorEngine()

    async def test_deterministic_execution_flow(self):
        ctx = PromptContext(
            user_id="integration_user_1",
            session_id="sess_det_1",
            initial_prompt="calculate 50 * 2"
        )

        response = await self.engine.run(ctx)
        self.assertEqual(response.status, "COMPLETE")
        self.assertEqual(response.route.route_type, "deterministic")
        self.assertTrue(len(response.executed_tools) >= 1)
        self.assertIn("100", response.final_output)

        # Verify state persistence
        history = await state_repository.get_session_context("sess_det_1")
        self.assertTrue(len(history) >= 2)

    async def test_multi_turn_tool_loop(self):
        ctx = PromptContext(
            user_id="integration_user_2",
            session_id="sess_loop_1",
            initial_prompt="What is the weather in London right now?"
        )

        response = await self.engine.run(ctx)
        self.assertEqual(response.status, "COMPLETE")
        self.assertTrue(response.iterations <= self.engine.max_depth)


if __name__ == "__main__":
    unittest.main()

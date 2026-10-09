"""Performance & Latency Benchmark Tests."""

import time
import unittest
from app.orchestrator.engine import OrchestratorEngine
from app.schemas.context import PromptContext


class TestPerformance(unittest.IsolatedAsyncioTestCase):
    """Benchmarks dispatch latency and execution overhead."""

    def setUp(self):
        self.engine = OrchestratorEngine()

    async def test_deterministic_route_latency(self):
        """Ensures deterministic fast routing executes with low latency."""
        ctx = PromptContext(
            user_id="perf_user_1",
            session_id="perf_sess_1",
            initial_prompt="system info"
        )

        start = time.perf_counter()
        response = await self.engine.run(ctx)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.assertEqual(response.status, "COMPLETE")
        self.assertLess(elapsed_ms, 750.0)


if __name__ == "__main__":
    unittest.main()

"""Unit Tests for Review-to-Routing Feedback Loop, Token Intelligence Engine, and Dashboard Aggregates."""

import unittest
import asyncio
from app.storage.repository import state_repository
from app.services.token_planner import token_planner, TokenBudgetPlan
from app.services.recommender import model_recommender
from app.orchestrator.router import intent_router
from app.services.catalogue import model_catalogue


class TestFeedbackAndTokenEngine(unittest.TestCase):

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_bayesian_smoothed_rating_calculation(self):
        """Validates Bayesian smoothed rating: (n * obs + m * prior) / (n + m)."""
        async def _test():
            import uuid
            test_mid = f"test-engine-{uuid.uuid4().hex[:6]}"
            email = f"rating_tester_{uuid.uuid4().hex[:6]}@openpilot.ai"
            user = await state_repository.create_user(email, "Rating Tester", "ValidPassword123!")
            
            # Prior: m = 5.0, prior_avg = 4.0
            # 1 rating of 5.0 -> (1*5 + 5*4) / 6 = 25/6 = 4.17
            await state_repository.save_or_update_review(
                user_id=user.id,
                session_id="test_sess_rate_1",
                model_id=test_mid,
                provider="Ollama (Local)",
                task_category="coding",
                rating=5,
                is_correct=True,
                is_useful=True,
                is_complete=True,
                is_fast=True,
                cost_acceptable=True
            )

            stats = await state_repository.get_model_task_rating_stats(test_mid, "coding")
            self.assertEqual(stats["review_count"], 1)
            self.assertEqual(stats["observed_avg"], 5.0)
            self.assertAlmostEqual(stats["smoothed_rating"], 4.17, places=2)
            self.assertEqual(stats["aspects"]["correct_pct"], 100.0)

            # Update the same session review to 2 stars (verifying duplicate prevention and upsert)
            updated = await state_repository.save_or_update_review(
                user_id=user.id,
                session_id="test_sess_rate_1",
                model_id=test_mid,
                provider="Ollama (Local)",
                task_category="coding",
                rating=2,
                is_correct=False
            )
            self.assertEqual(updated["action"], "updated")

            stats2 = await state_repository.get_model_task_rating_stats(test_mid, "coding")
            self.assertEqual(stats2["review_count"], 1)
            self.assertEqual(stats2["observed_avg"], 2.0)
            # (1*2 + 5*4) / 6 = 22/6 = 3.67
            self.assertAlmostEqual(stats2["smoothed_rating"], 3.67, places=2)

        self.loop.run_until_complete(_test())

    def test_token_budget_planner_allocation(self):
        """Validates token budget plan adapts output tier, decomposition, and reserves based on task."""
        spec = model_catalogue.get_model("gemma-4-e4b")
        
        # Coding task
        code_plan = token_planner.plan_budget(
            prompt="def quicksort(arr): write unit tests",
            history=[],
            task_category="coding",
            model_spec=spec
        )
        self.assertEqual(code_plan.output_budget_tier, "standard_code")
        self.assertEqual(code_plan.allocated_output_tokens, 1200)
        self.assertGreater(code_plan.baseline_tokens_estimate, code_plan.total_planned_tokens)
        self.assertGreater(code_plan.projected_savings_pct, 0.0)

        # Tool execution task
        tool_plan = token_planner.plan_budget(
            prompt="what is 50 * 20?",
            history=[],
            task_category="tool_execution",
            model_spec=spec
        )
        self.assertEqual(tool_plan.output_budget_tier, "concise")
        self.assertEqual(tool_plan.decomposition_strategy, "model_plus_deterministic_tools")

    def test_token_accounting_measured_savings(self):
        """Validates token accounting calculates exact tokens saved against baseline."""
        spec = model_catalogue.get_model("gemma-4-e4b")
        plan = token_planner.plan_budget(
            prompt="Short test prompt",
            history=[],
            task_category="general",
            model_spec=spec
        )

        accounting = token_planner.compute_accounting(
            plan=plan,
            actual_input_tokens=15,
            actual_output_tokens=60,
            verification_tokens=40,
            model_spec=spec
        )
        self.assertEqual(accounting.total_tokens, 15 + 60 + 40)
        self.assertGreater(accounting.baseline_tokens, accounting.total_tokens)
        self.assertEqual(accounting.tokens_saved, accounting.baseline_tokens - accounting.total_tokens)
        self.assertGreater(accounting.savings_percentage, 0.0)

    def test_recommender_adjusts_by_user_feedback(self):
        """Validates recommender boosts well-reviewed models and penalizes poorly-reviewed models."""
        understanding = intent_router.understand_request("write a python binary search algorithm")

        # Ratings mocking: Qwen coder has 4.9 smoothed rating (boosted), DeepSeek R1 has 2.0 (penalized)
        mock_ratings = {
            "qwen2.5-coder:7b": {"smoothed_rating": 4.9, "review_count": 8},
            "deepseek-r1:8b": {"smoothed_rating": 2.0, "review_count": 5}
        }

        rec = model_recommender.recommend(
            understanding,
            ranking_mode="best_performance",
            ratings=mock_ratings
        )
        # Verify recommended candidates have rating displays
        qwen = next((m for m in rec.models if m.model_id == "qwen2.5-coder:7b"), None)
        self.assertIsNotNone(qwen)
        self.assertIn("4.9★", qwen.rating_display)

    def test_dashboard_aggregates(self):
        """Validates dashboard stats return non-null metrics derived from database."""
        async def _test():
            stats = await state_repository.get_dashboard_aggregates()
            self.assertIn("total_requests", stats)
            self.assertIn("success_rate_pct", stats)
            self.assertIn("model_usage_distribution", stats)
            self.assertIn("category_distribution", stats)
            self.assertIn("free_vs_paid", stats)
            self.assertIn("total_tokens_consumed", stats)
            self.assertIn("estimated_tokens_saved", stats)
            self.assertIn("latency", stats)
            self.assertIn("average_rating", stats)

        self.loop.run_until_complete(_test())


if __name__ == "__main__":
    unittest.main()

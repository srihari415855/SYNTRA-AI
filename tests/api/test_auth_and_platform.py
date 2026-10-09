"""Comprehensive API Tests for Authentication, Understanding, Recommendations, and Execution."""

import unittest
from fastapi.testclient import TestClient
from app.main import app


class TestAuthAndPlatform(unittest.TestCase):
    """Verifies authentication, dynamic understanding, catalogue, and ranking policies."""

    def setUp(self):
        self.client = TestClient(app)

    def test_demo_login_flow(self):
        """Tests instant sandboxed guest/demo session generation."""
        res = self.client.post("/api/v1/auth/demo")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("access_token", data)
        self.assertTrue(data["is_demo"])
        self.assertEqual(data["email"], "demo@openpilot.ai")

    def test_user_registration_and_login_flow(self):
        """Tests full registration, validation, and subsequent login flow."""
        email = f"testuser_{id(self)}@openpilot.ai"
        reg_payload = {
            "email": email,
            "name": "Jane Architect",
            "password": "StrongPassword2026!"
        }
        # 1. Register
        reg_res = self.client.post("/api/v1/auth/register", json=reg_payload)
        self.assertEqual(reg_res.status_code, 201)
        reg_data = reg_res.json()
        self.assertIn("access_token", reg_data)
        self.assertEqual(reg_data["email"], email)

        # 2. Login
        login_res = self.client.post("/api/v1/auth/login", json={
            "email": email,
            "password": "StrongPassword2026!"
        })
        self.assertEqual(login_res.status_code, 200)
        login_data = login_res.json()
        self.assertIn("access_token", login_data)

    def test_registration_weak_password_rejection(self):
        """Rejects passwords shorter than 8 characters or without special characters/digits."""
        res = self.client.post("/api/v1/auth/register", json={
            "email": "weak@example.com",
            "name": "Weak Pwd",
            "password": "short"
        })
        self.assertIn(res.status_code, (400, 422))

    def test_dynamic_request_understanding(self):
        """Tests classification of coding and math tasks."""
        # Coding
        code_res = self.client.post("/api/v1/understand", json={
            "prompt": "def quicksort(arr): return arr",
            "has_image": False,
            "has_audio": False
        })
        self.assertEqual(code_res.status_code, 200)
        self.assertEqual(code_res.json()["task_category"], "coding")

        # Vision
        vision_res = self.client.post("/api/v1/understand", json={
            "prompt": "What is in this diagram?",
            "has_image": True,
            "has_audio": False
        })
        self.assertEqual(vision_res.status_code, 200)
        self.assertEqual(vision_res.json()["task_category"], "image_understanding")

    def test_dynamic_recommendations_5_modes(self):
        """Tests that recommendations evaluate across all 5 ranking modes."""
        modes = ["best_value", "best_performance", "lowest_price", "free", "paid"]
        for mode in modes:
            res = self.client.post("/api/v1/recommend", json={
                "prompt": "Write a Python script to parse JSON logs",
                "ranking_mode": mode
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["ranking_mode"], mode)
            self.assertGreaterEqual(len(data["models"]), 1)
            # Verify top pick is identified
            self.assertTrue(data["models"][0]["is_top_pick"])

    def test_model_catalogue_endpoint(self):
        """Tests retrieval of 10 verified models with metadata."""
        res = self.client.get("/api/v1/catalogue")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_models"], 10)
        model_ids = [m["id"] for m in data["models"]]
        self.assertIn("gemma-4-e4b", model_ids)
        self.assertIn("claude-3-5-sonnet", model_ids)
        self.assertIn("gpt-4o", model_ids)

    def test_real_model_execution_endpoint(self):
        """Tests execution of math prompt producing calculated outcome."""
        res = self.client.post("/api/v1/execute", json={
            "prompt": "calculate 35 * 4",
            "selected_model": "gemma-4-e4b",
            "ranking_mode": "best_value"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "COMPLETE")
        self.assertIn("140", data["final_output"])
        self.assertGreater(data["execution_time_ms"], 0)


if __name__ == "__main__":
    unittest.main()

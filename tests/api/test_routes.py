"""API Tests for FastAPI Ingress Endpoints."""

import unittest
from fastapi.testclient import TestClient
from app.main import app


class TestAPIRoutes(unittest.TestCase):
    """Verifies REST API endpoints."""

    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertGreaterEqual(data["registered_tools"], 3)

    def test_list_tools(self):
        response = self.client.get("/api/v1/tools")
        self.assertEqual(response.status_code, 200)
        tools = response.json()
        tool_names = [t["name"] for t in tools]
        self.assertIn("calculator", tool_names)
        self.assertIn("system_info", tool_names)
        self.assertIn("weather_lookup", tool_names)

    def test_chat_endpoint_valid_request(self):
        payload = {
            "user_id": "test_user_api",
            "prompt": "Calculate 12 * 12"
        }
        response = self.client.post("/api/v1/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "COMPLETE")
        self.assertIn("144", data["final_output"])


if __name__ == "__main__":
    unittest.main()

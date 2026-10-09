"""Unit Tests for Tool Registry and Execution Runtime."""

import unittest
from app.schemas.tool_call import ToolCall
from app.tools.definitions import calculator, system_info, weather_lookup
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


class TestTools(unittest.IsolatedAsyncioTestCase):
    """Verifies tool definitions, argument validation, and sandboxed executor."""

    def setUp(self):
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(registry=self.registry)

    def test_calculator_direct(self):
        res = calculator("25 * 4 + 10")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["result"], 110)

    def test_calculator_invalid_expression(self):
        res = calculator("import os; os.system('echo hi')")
        self.assertEqual(res["status"], "error")

    def test_system_info_direct(self):
        res = system_info()
        self.assertIn("os", res)
        self.assertEqual(res["status"], "healthy")

    def test_weather_direct(self):
        res = weather_lookup("London")
        self.assertEqual(res["location"], "London")
        self.assertIn("temperature_celsius", res)

    async def test_executor_successful_call(self):
        call = ToolCall(id="c1", name="calculator", arguments={"expression": "100 / 4"})
        res = await self.executor.execute(call)
        self.assertFalse(res.is_error)
        self.assertEqual(res.output["result"], 25.0)

    async def test_executor_missing_argument(self):
        call = ToolCall(id="c2", name="calculator", arguments={})
        res = await self.executor.execute(call)
        self.assertTrue(res.is_error)
        self.assertIn("Missing required argument", res.error_message)

    async def test_executor_unknown_tool(self):
        call = ToolCall(id="c3", name="non_existent_tool", arguments={})
        res = await self.executor.execute(call)
        self.assertTrue(res.is_error)
        self.assertIn("not registered", res.error_message)


if __name__ == "__main__":
    unittest.main()

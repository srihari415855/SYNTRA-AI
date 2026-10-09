"""Unit Tests for Core Pydantic Data Contracts."""

import unittest
from pydantic import ValidationError
from app.schemas.context import PromptContext
from app.schemas.tool_call import ToolDefinition, ToolCall, ToolResult
from app.schemas.agent_state import AgentState
from app.schemas.responses import RouteDecision, ModelResponse


class TestCoreSchemas(unittest.TestCase):
    """Verifies schema contracts defined in Part 3 of MVP specification."""

    def test_prompt_context_validation(self):
        ctx = PromptContext(
            user_id="user_123",
            initial_prompt="Calculate 10 + 20",
            history=[{"role": "user", "content": "hello"}]
        )
        self.assertEqual(ctx.user_id, "user_123")
        self.assertEqual(len(ctx.history), 1)

    def test_prompt_context_missing_fields(self):
        with self.assertRaises(ValidationError):
            PromptContext.model_validate({"user_id": "user_123"})

    def test_tool_definition(self):
        tool = ToolDefinition(
            name="calculator",
            description="Evaluates math",
            parameters={"type": "object", "properties": {"expression": {"type": "string"}}}
        )
        self.assertEqual(tool.name, "calculator")

    def test_agent_state_defaults(self):
        state = AgentState(
            session_id="session_abc",
            current_context=[]
        )
        self.assertEqual(state.final_status, "IN_PROGRESS")
        self.assertEqual(state.iteration_count, 0)
        self.assertIsNone(state.last_tool_call)


if __name__ == "__main__":
    unittest.main()

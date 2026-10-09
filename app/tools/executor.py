"""Tool Runtime Execution and Sandbox Engine."""

import asyncio
import inspect
import time
from typing import Any, Dict, Optional

from app.core.config import settings
from app.core.exceptions import (
    ToolArgumentValidationError,
    ToolExecutionError,
    ToolNotFoundError,
)
from app.core.logging import get_logger
from app.schemas.tool_call import ToolCall, ToolResult
from app.tools.registry import ToolRegistry, tool_registry

logger = get_logger("ai_system.executor")


class ToolExecutor:
    """
    Isolated execution runtime for executing registered tools with strict argument validation,
    bounded timeouts, and error trapping.
    """

    def __init__(self, registry: Optional[ToolRegistry] = None, timeout: Optional[float] = None):
        self.registry = registry or tool_registry
        self.timeout = timeout or settings.TOOL_EXECUTION_TIMEOUT_SECONDS

    def validate_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """
        Validates arguments against the registered JSON schema.
        Raises ToolArgumentValidationError if required fields are missing.
        """
        definition = self.registry.get_definition(tool_name)
        if not definition:
            raise ToolNotFoundError(tool_name)

        params_schema = definition.parameters or {}
        required_fields = params_schema.get("required", [])

        missing_fields = [f for f in required_fields if f not in arguments]
        if missing_fields:
            raise ToolArgumentValidationError(
                tool_name=tool_name,
                errors=[f"Missing required argument(s): {', '.join(missing_fields)}"]
            )

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Executes a single tool call asynchronously within a bounded timeout.
        Returns a structured ToolResult object. Never crashes the orchestrator.
        """
        start_time = time.perf_counter()
        tool_name = tool_call.name
        call_id = tool_call.id
        arguments = tool_call.arguments or {}

        logger.info(
            f"Executing tool '{tool_name}'",
            extra={"event_type": "TOOL_EXEC_START", "tool_name": tool_name}
        )

        # 1. Verify existence
        if not self.registry.has_tool(tool_name):
            duration_ms = (time.perf_counter() - start_time) * 1000
            err_msg = f"Tool '{tool_name}' is not registered or permitted."
            logger.error(err_msg, extra={"event_type": "TOOL_NOT_FOUND", "tool_name": tool_name})
            return ToolResult(
                tool_name=tool_name,
                call_id=call_id,
                output=None,
                is_error=True,
                error_message=err_msg,
                execution_time_ms=round(duration_ms, 2)
            )

        # 2. Validate arguments
        try:
            self.validate_arguments(tool_name, arguments)
        except ToolArgumentValidationError as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                f"Validation error for '{tool_name}': {exc}",
                extra={"event_type": "TOOL_ARG_VALIDATION_ERROR", "tool_name": tool_name}
            )
            return ToolResult(
                tool_name=tool_name,
                call_id=call_id,
                output=None,
                is_error=True,
                error_message=str(exc),
                execution_time_ms=round(duration_ms, 2)
            )

        # 3. Execute callable with timeout
        func = self.registry.get_tool(tool_name)
        try:
            if inspect.iscoroutinefunction(func):
                task = func(**arguments)
            else:
                loop = asyncio.get_running_loop()
                task = loop.run_in_executor(None, lambda: func(**arguments))

            raw_result = await asyncio.wait_for(task, timeout=self.timeout)
            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                f"Successfully completed tool '{tool_name}' in {duration_ms:.2f}ms",
                extra={"event_type": "TOOL_EXEC_SUCCESS", "tool_name": tool_name, "latency_ms": duration_ms}
            )
            return ToolResult(
                tool_name=tool_name,
                call_id=call_id,
                output=raw_result,
                is_error=False,
                error_message=None,
                execution_time_ms=round(duration_ms, 2)
            )

        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            err_msg = f"Execution of tool '{tool_name}' timed out after {self.timeout} seconds."
            logger.error(err_msg, extra={"event_type": "TOOL_TIMEOUT", "tool_name": tool_name})
            return ToolResult(
                tool_name=tool_name,
                call_id=call_id,
                output=None,
                is_error=True,
                error_message=err_msg,
                execution_time_ms=round(duration_ms, 2)
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            err_msg = f"Runtime error in tool '{tool_name}': {exc}"
            logger.error(err_msg, extra={"event_type": "TOOL_ERROR", "tool_name": tool_name, "error": str(exc)})
            return ToolResult(
                tool_name=tool_name,
                call_id=call_id,
                output=None,
                is_error=True,
                error_message=err_msg,
                execution_time_ms=round(duration_ms, 2)
            )


tool_executor = ToolExecutor()

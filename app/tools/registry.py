"""Tool Registry Module."""

import inspect
from typing import Any, Callable, Dict, List, Optional
from app.schemas.tool_call import ToolDefinition
from app.tools.definitions import calculator, system_info, weather_lookup


class ToolRegistry:
    """Central repository storing metadata and executable callables for all available tools."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._definitions: Dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def register(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """Decorator to register a Python callable as a recognized tool."""
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or (func.__doc__ or "").strip().split("\n\n")[0].strip()
            tool_params = parameters or self._infer_schema_from_signature(func)

            definition = ToolDefinition(
                name=tool_name,
                description=tool_desc,
                parameters=tool_params
            )
            self._tools[tool_name] = func
            self._definitions[tool_name] = definition
            return func
        return decorator

    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """Direct programmatic registration of a callable."""
        self.register(name=name, description=description, parameters=parameters)(func)

    def _infer_schema_from_signature(self, func: Callable) -> Dict[str, Any]:
        """Infers OpenAPI/JSON schema from Python function annotations."""
        sig = inspect.signature(func)
        properties: Dict[str, Any] = {}
        required: List[str] = []

        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            param_type = type_map.get(param.annotation, "string")
            properties[param_name] = {
                "type": param_type,
                "description": f"Argument {param_name}"
            }
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    def _register_default_tools(self) -> None:
        """Registers the built-in MVP tools."""
        self.register_function(
            calculator,
            name="calculator",
            description="Evaluates mathematical and arithmetic expressions safely.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A mathematical string expression (e.g., '14 * 5 + 20')."
                    }
                },
                "required": ["expression"]
            }
        )
        self.register_function(
            system_info,
            name="system_info",
            description="Retrieves system diagnostics, operating system details, and Python environment info.",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
        self.register_function(
            weather_lookup,
            name="weather_lookup",
            description="Fetches real-time weather conditions and forecasts for a specified city or location.",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or region (e.g. 'London', 'Tokyo')."
                    }
                },
                "required": ["location"]
            }
        )

    def get_tool(self, name: str) -> Optional[Callable]:
        """Returns the callable for a tool name."""
        return self._tools.get(name)

    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        """Returns the ToolDefinition schema for a tool name."""
        return self._definitions.get(name)

    def list_definitions(self) -> List[ToolDefinition]:
        """Returns all registered tool schemas."""
        return list(self._definitions.values())

    def has_tool(self, name: str) -> bool:
        """Checks if a tool is registered."""
        return name in self._tools


tool_registry = ToolRegistry()

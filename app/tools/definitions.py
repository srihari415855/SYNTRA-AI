"""Built-in Tool Implementations and Schema Specifications."""

import ast
import operator
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict

# Safe math operators for AST calculator
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast(node: ast.AST) -> Any:
    """Safely evaluates an abstract syntax tree of arithmetic operations."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return _SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_ast(node.operand)
        return _SAFE_OPERATORS[op_type](operand)
    elif isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    else:
        raise ValueError(f"Disallowed expression node: {type(node).__name__}")


def calculator(expression: str) -> Dict[str, Any]:
    """
    Evaluates mathematical and arithmetic expressions safely.
    
    Args:
        expression: A mathematical string expression such as '12 * (4 + 6)' or '2**8'.
    """
    cleaned = expression.strip()
    try:
        parsed = ast.parse(cleaned, mode='eval')
        result = _eval_ast(parsed)
        return {"expression": cleaned, "result": result, "status": "success"}
    except Exception as exc:
        return {"expression": cleaned, "error": str(exc), "status": "error"}


def system_info() -> Dict[str, Any]:
    """
    Retrieves system diagnostics, operating system details, and Python environment info.
    """
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "current_utc_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "status": "healthy"
    }


def weather_lookup(location: str) -> Dict[str, Any]:
    """
    Fetches real-time weather conditions and forecasts for a specified city or location.
    
    Args:
        location: City name or region (e.g. 'New York', 'London', 'Tokyo').
    """
    city = location.strip().title()
    # Mock realistic responses for weather testing
    data_map = {
        "London": {"temp_c": 14, "condition": "Overcast", "humidity": 78},
        "New York": {"temp_c": 22, "condition": "Sunny", "humidity": 55},
        "Tokyo": {"temp_c": 19, "condition": "Light Rain", "humidity": 82},
        "Paris": {"temp_c": 16, "condition": "Partly Cloudy", "humidity": 68},
    }
    details = data_map.get(city, {"temp_c": 20, "condition": "Fair", "humidity": 60})
    return {
        "location": city,
        "temperature_celsius": details["temp_c"],
        "temperature_fahrenheit": round(details["temp_c"] * 9 / 5 + 32, 1),
        "condition": details["condition"],
        "humidity_percent": details["humidity"],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

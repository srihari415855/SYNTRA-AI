"""Custom Domain Exceptions for the AI Model-Routing System."""


class AISystemBaseException(Exception):
    """Base exception for all domain-specific errors."""
    pass


class ModelInferenceError(AISystemBaseException):
    """Raised when an external model inference call fails."""
    def __init__(self, message: str, model_name: str, status_code: int = 500):
        super().__init__(message)
        self.model_name = model_name
        self.status_code = status_code


class ModelRateLimitError(ModelInferenceError):
    """Raised when rate limits or HTTP 429 are encountered."""
    pass


class ModelUnavailableError(ModelInferenceError):
    """Raised when the model runtime (e.g. Ollama daemon) is offline or unreachable."""
    pass


class ToolExecutionError(AISystemBaseException):
    """Raised when a registered tool fails during runtime execution."""
    def __init__(self, message: str, tool_name: str, is_transient: bool = False):
        super().__init__(message)
        self.tool_name = tool_name
        self.is_transient = is_transient


class ToolNotFoundError(AISystemBaseException):
    """Raised when the LLM requests a tool not in the registry."""
    def __init__(self, tool_name: str):
        super().__init__(f"Tool '{tool_name}' is not registered or permitted.")
        self.tool_name = tool_name


class ToolArgumentValidationError(AISystemBaseException):
    """Raised when tool arguments fail strict Pydantic validation."""
    def __init__(self, tool_name: str, errors: list):
        super().__init__(f"Invalid arguments for tool '{tool_name}': {errors}")
        self.tool_name = tool_name
        self.errors = errors


class MaxIterationDepthExceededError(AISystemBaseException):
    """Raised when the orchestrator loop exceeds max call depth."""
    def __init__(self, max_depth: int):
        super().__init__(f"Orchestration loop exceeded maximum operational depth threshold of {max_depth}.")
        self.max_depth = max_depth


class StatePersistenceError(AISystemBaseException):
    """Raised when transactional database commit fails."""
    pass

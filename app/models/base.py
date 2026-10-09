"""Abstract Base Classes (ABCs) Defining Core Architectural Contracts."""

import abc
from typing import Any, Dict, List, Optional
from app.schemas.responses import ValidationReport, ModelResponse


class ModelConnectorABC(abc.ABC):
    """
    Abstract Base Class defining required methods for all Large Language Model connectors.
    Ensures any model implementation (Ollama, OpenAI, Anthropic, or Mock) adheres to a common interface.
    """

    @abc.abstractmethod
    def connect(self, api_key: Optional[str] = None) -> bool:
        """
        Initializes the connection using an API key or base URL and validates availability.
        """
        raise NotImplementedError("Subclasses must implement the 'connect' method.")

    @abc.abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 1024) -> Optional[str]:
        """
        Sends a natural language prompt to the model and returns the raw generated text response.
        """
        raise NotImplementedError("Subclasses must implement the 'generate' method.")

    @abc.abstractmethod
    async def generate_structured(
        self,
        messages: List[Dict[str, Any]],
        tool_schemas: List[Dict[str, Any]],
        model_name: Optional[str] = None
    ) -> ModelResponse:
        """
        Specialized generation call for function/tool calling.
        Returns a structured ModelResponse containing either tool_calls or final content.
        """
        raise NotImplementedError("Subclasses must implement structured generation.")


class InputValidatorABC(abc.ABC):
    """
    Abstract Base Class defining the contract for validating incoming data payloads before processing.
    Mitigates Schema Drift and malformed user input.
    """

    @abc.abstractmethod
    def validate(self, data: Dict[str, Any]) -> ValidationReport:
        """
        Performs comprehensive structural and semantic validation on a given data payload.
        """
        raise NotImplementedError("Subclasses must implement the 'validate' method.")


class OutputParserABC(abc.ABC):
    """
    Abstract Base Class defining the contract for converting raw LLM textual output into structured data.
    """

    @abc.abstractmethod
    def parse(self, raw_output: str) -> Dict[str, Any]:
        """
        Interprets and structures raw textual or JSON output into standard dictionary payloads.
        """
        raise NotImplementedError("Subclasses must implement the 'parse' method.")

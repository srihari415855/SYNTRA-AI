"""Model Registry and Connector Management."""

from typing import Dict, Optional
from app.core.config import settings
from app.models.base import ModelConnectorABC
from app.models.gemini_adapter import GeminiAdapter
from app.models.ollama_adapter import OllamaAdapter


class ModelRegistry:
    """Registry maintaining active model connectors and model aliases."""

    def __init__(self):
        self._connectors: Dict[str, ModelConnectorABC] = {}
        self._default_connector: Optional[ModelConnectorABC] = None
        self._init_default_connector()

    def _init_default_connector(self) -> None:
        """Initializes the primary Ollama and Gemini adapter instances."""
        ollama_adapter = OllamaAdapter(
            base_url=settings.OLLAMA_BASE_URL,
            default_model=settings.DEFAULT_MODEL,
            fallback_enabled=settings.USE_MOCK_FALLBACK_ON_OLLAMA_OFFLINE
        )
        self.register_connector("ollama", ollama_adapter, is_default=True)

        gemini_adapter = GeminiAdapter(
            api_key=getattr(settings, "GEMINI_API_KEY", None),
            default_model="gemini-2.0-flash"
        )
        self.register_connector("gemini", gemini_adapter, is_default=False)

    def register_connector(self, name: str, connector: ModelConnectorABC, is_default: bool = False) -> None:
        """Registers a connector instance."""
        self._connectors[name] = connector
        if is_default or self._default_connector is None:
            self._default_connector = connector

    def get_connector(self, name: Optional[str] = None) -> ModelConnectorABC:
        """Retrieves a named connector or the default instance."""
        if name and name in self._connectors:
            return self._connectors[name]
        if self._default_connector:
            return self._default_connector
        raise ValueError("No model connector registered in registry.")

    def get_connector_for_model(self, model_name: Optional[str] = None) -> ModelConnectorABC:
        """Intelligently routes model name to its designated provider adapter."""
        if not model_name:
            return self.get_connector()

        low = model_name.lower()
        if "gemini" in low:
            return self.get_connector("gemini")
        return self.get_connector("ollama")

    def get_gemini_adapter(self) -> Optional[GeminiAdapter]:
        """Convenience method to access GeminiAdapter instance."""
        adapter = self._connectors.get("gemini")
        if isinstance(adapter, GeminiAdapter):
            return adapter
        return None


model_registry = ModelRegistry()


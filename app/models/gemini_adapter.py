"""Google Gemini Model Connector and Adapter Implementation.

Supports Gemini 2.0 Flash, Gemini 1.5 Pro, and Gemini 1.5 Flash via the official
google.genai SDK with automated multimodal and tool-execution support.
"""

import base64
import os
import uuid
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.exceptions import (
    ModelInferenceError,
    ModelRateLimitError,
    ModelUnavailableError,
)
from app.core.logging import get_logger
from app.models.base import ModelConnectorABC
from app.schemas.responses import ModelResponse
from app.schemas.tool_call import ToolCall

logger = get_logger("ai_system.gemini")

# Canonical Gemini Model Aliases
GEMINI_MODELS = {
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-1.5-pro": "gemini-1.5-pro",
}


class GeminiAdapter(ModelConnectorABC):
    """
    Adapter implementing ModelConnectorABC for Google Gemini frontier engines.
    Connects to Gemini via google.genai SDK with automatic multimodal conversion.
    """

    def __init__(self, api_key: Optional[str] = None, default_model: str = "gemini-2.0-flash"):
        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        self.default_model = default_model
        self._client = None
        self._is_connected = False
        if self._api_key:
            self._init_client()

    def _init_client(self) -> None:
        """Initializes genai.Client if API key is present."""
        if not self._api_key:
            return
        try:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
            self._is_connected = True
        except Exception as exc:
            logger.error(f"Failed to initialize Google GenAI client: {exc}")
            self._client = None
            self._is_connected = False

    def set_api_key(self, api_key: str) -> bool:
        """Updates the active API key and re-tests connectivity."""
        self._api_key = api_key.strip()
        self._init_client()
        return self.connect(self._api_key)

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key and self._client)

    def connect(self, api_key: Optional[str] = None) -> bool:
        """Verifies connectivity to Google Gemini API."""
        key_to_test = api_key or self._api_key
        if not key_to_test:
            self._is_connected = False
            return False

        try:
            from google import genai
            test_client = genai.Client(api_key=key_to_test)
            # Lightweight probe
            resp = test_client.models.generate_content(
                model=self.default_model,
                contents="ping"
            )
            self._api_key = key_to_test
            self._client = test_client
            self._is_connected = bool(resp and resp.text)
            return self._is_connected
        except Exception as exc:
            logger.warning(f"Gemini connection probe failed: {exc}")
            self._is_connected = False
            return False

    async def generate(self, prompt: str, max_tokens: int = 1024) -> Optional[str]:
        """Sends single prompt to Gemini."""
        messages = [{"role": "user", "content": prompt}]
        response = await self.generate_structured(messages=messages, tool_schemas=[])
        return response.content

    async def generate_structured(
        self,
        messages: List[Dict[str, Any]],
        tool_schemas: List[Dict[str, Any]],
        model_name: Optional[str] = None
    ) -> ModelResponse:
        """
        Executes generation against Google Gemini models with multimodal context.
        """
        target_model = model_name or self.default_model
        if target_model.startswith("models/"):
            target_model = target_model.replace("models/", "")

        # Check API key presence
        if not self._api_key or not self._client:
            raise ModelUnavailableError(
                f"Google Gemini API Key is required to run '{target_model}'. "
                "Please configure GEMINI_API_KEY in Settings or .env file.",
                model_name=target_model
            )

        try:
            from google.genai import types

            # Build Gemini contents from messages
            contents: List[Any] = []
            system_instruction: Optional[str] = None

            for m in messages:
                role = m.get("role", "user")
                text_content = str(m.get("content", ""))
                image_b64 = m.get("image_data")

                if role == "system":
                    system_instruction = text_content
                    continue

                gemini_role = "user" if role in ("user", "tool") else "model"
                parts: List[Any] = []

                # Attach image if present
                if image_b64 and "base64," in image_b64:
                    try:
                        header, b64_data = image_b64.split("base64,", 1)
                        mime_type = "image/png"
                        if "jpeg" in header or "jpg" in header:
                            mime_type = "image/jpeg"
                        elif "webp" in header:
                            mime_type = "image/webp"

                        raw_bytes = base64.b64decode(b64_data)
                        parts.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime_type))
                    except Exception as img_err:
                        logger.warning(f"Could not parse image payload for Gemini: {img_err}")

                if text_content:
                    parts.append(text_content)

                if parts:
                    contents.append(types.Content(role=gemini_role, parts=parts))

            # Build config
            config_params: Dict[str, Any] = {}
            if system_instruction:
                config_params["system_instruction"] = system_instruction

            # Execute generation
            config = types.GenerateContentConfig(**config_params) if config_params else None

            # Asynchronous execution
            response = self._client.models.generate_content(
                model=target_model,
                contents=contents,
                config=config
            )

            response_text = response.text or ""
            prompt_tokens = 0
            completion_tokens = 0

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

            return ModelResponse(
                content=response_text,
                tool_calls=[],
                model_name=target_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason="stop",
                raw_response={"text": response_text}
            )

        except Exception as exc:
            err_msg = str(exc)
            logger.error(f"Gemini inference failed on {target_model}: {err_msg}")
            if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                raise ModelRateLimitError(
                    f"Gemini API rate limit exceeded: {err_msg}",
                    model_name=target_model,
                    status_code=429
                ) from exc
            if "API_KEY_INVALID" in err_msg or "UNAUTHENTICATED" in err_msg:
                raise ModelUnavailableError(
                    f"Invalid Gemini API key: {err_msg}",
                    model_name=target_model
                ) from exc
            raise ModelInferenceError(
                f"Google Gemini inference error: {err_msg}",
                model_name=target_model
            ) from exc

"""Application Configuration Module."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file from ai-system root if present
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()


class Settings(BaseModel):
    """Strongly typed application configuration settings."""

    # Environment
    ENVIRONMENT: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    DEBUG: bool = Field(default_factory=lambda: os.getenv("DEBUG", "true").lower() in ("true", "1", "yes"))

    # API Server
    API_HOST: str = Field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    API_PORT: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    API_TITLE: str = Field(default_factory=lambda: os.getenv("API_TITLE", "AI Model-Routing System"))
    API_VERSION: str = Field(default_factory=lambda: os.getenv("API_VERSION", "0.1.0"))

    # Local LLM Runtime (Ollama / Local HTTP)
    OLLAMA_BASE_URL: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    DEFAULT_MODEL: str = Field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "gemma-4-e4b"))
    CODING_MODEL: str = Field(default_factory=lambda: os.getenv("CODING_MODEL", "qwen2.5-coder:7b"))
    REASONING_MODEL: str = Field(default_factory=lambda: os.getenv("REASONING_MODEL", "qwen3:8b"))
    COMPLEX_REASONING_MODEL: str = Field(default_factory=lambda: os.getenv("COMPLEX_REASONING_MODEL", "deepseek-r1:8b"))

    # Google Gemini Cloud Integration
    GEMINI_API_KEY: Optional[str] = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    )

    # Fallback & Mocking
    USE_MOCK_FALLBACK_ON_OLLAMA_OFFLINE: bool = Field(
        default_factory=lambda: os.getenv("USE_MOCK_FALLBACK_ON_OLLAMA_OFFLINE", "true").lower() in ("true", "1", "yes")
    )
    OLLAMA_TIMEOUT_SECONDS: float = Field(
        default_factory=lambda: float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120.0"))
    )

    # Orchestrator & Loop Guards
    MAX_CALL_DEPTH: int = Field(default_factory=lambda: int(os.getenv("MAX_CALL_DEPTH", "5")))
    TOOL_EXECUTION_TIMEOUT_SECONDS: float = Field(
        default_factory=lambda: float(os.getenv("TOOL_EXECUTION_TIMEOUT_SECONDS", "15.0"))
    )

    # Tenacity Retries
    RETRY_MAX_ATTEMPTS: int = Field(default_factory=lambda: int(os.getenv("RETRY_MAX_ATTEMPTS", "3")))
    RETRY_MIN_WAIT_SECONDS: float = Field(
        default_factory=lambda: float(os.getenv("RETRY_MIN_WAIT_SECONDS", "1.0"))
    )
    RETRY_MAX_WAIT_SECONDS: float = Field(
        default_factory=lambda: float(os.getenv("RETRY_MAX_WAIT_SECONDS", "8.0"))
    )
    RETRY_JITTER_SECONDS: float = Field(
        default_factory=lambda: float(os.getenv("RETRY_JITTER_SECONDS", "1.0"))
    )

    # Database Persistence
    DATABASE_URL: str = Field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./ai_system.db")
    )
    SQL_ECHO: bool = Field(
        default_factory=lambda: os.getenv("SQL_ECHO", "false").lower() in ("true", "1", "yes")
    )


settings = Settings()

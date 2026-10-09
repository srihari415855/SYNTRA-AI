# Architectural Decision Records (ADRs)

| Area | Selected Decision | Justification |
| :--- | :--- | :--- |
| **API Gateway** | `FastAPI + Uvicorn` | Asynchronous, non-blocking HTTP ingress, native OpenAPI generation, automatic Pydantic validation. |
| **Data Contracts** | `Pydantic v2` | Strict compile-time and runtime validation, fast Rust-based core (`pydantic-core`), native JSON schema emission. |
| **Retry Handling** | `Tenacity` | Declarative, configurable exponential backoff with randomized jitter to handle transient model timeouts and HTTP 429s. |
| **Initial Local Runtime** | `Ollama` | Standardized REST API (`/api/chat`), straightforward model pulling, local privacy, and low deployment friction. |
| **Primary Model** | `Gemma 4 E4B` | Lightweight, highly responsive primary intent classification and general conversational reasoning. |
| **Specialist Models** | `Qwen2.5-Coder 7B`, `Qwen3 8B`, `DeepSeek-R1 8B` | Specialized task routing for programming queries, multi-step math/logic, and deep analytical reasoning. |
| **HTTP Client** | `HTTPX` (async) | Modern, non-blocking client with complete HTTP/1.1 and HTTP/2 support, timeouts, and test integration. |
| **Persistence** | `SQLite + SQLAlchemy 2.0` | Zero-configuration file-based relational storage with ACID guarantees, async I/O compatibility via aiosqlite, and easy migration path to PostgreSQL. |
| **Configuration** | `python-dotenv + Typed Config` | Clear environment separation, twelve-factor app principles, and no hardcoded credentials. |
| **Testing** | Standard `unittest` + `HTTPX TestClient` / `pytest` | Complete unit, integration, and API contract coverage without brittle external service dependencies. |
| **Logging** | Python `logging` with Structured JSON | Machine-readable audit trails for correlation IDs, latencies, model token counts, and tool execution. |

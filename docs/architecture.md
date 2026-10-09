# Architecture Specification: AI Model-Routing System MVP

## 1. Executive Summary
The AI Model-Routing System is an enterprise-grade function-calling and model-routing harness designed to bridge incoming user prompts, specialized local large language models (LLMs), registered Python tools, and transactional state persistence.

The system is decoupled into single-responsibility modules:
- **API Gateway (Ingress)**: FastAPI asynchronous REST layer.
- **Orchestrator**: Multi-turn execution control plane with circular loop depth guards (`max_depth = 5`).
- **Intent Router**: Two-tier routing combining deterministic pattern matching and model-based classification.
- **Model Interface Layer (MIL)**: Abstract adapter wrapping local runtime engines (Ollama) with Tenacity retry policies.
- **Tool Registry & Executor**: Secure schema introspection, Pydantic argument validation, timeout bounding, and error trapping.
- **State & Persistence Manager**: Transactional SQLite store persisting sessions, messages, and tool execution audit records.

---

## 2. Component Architecture Diagram

```
+-------------------------------------------------------------------------+
|                               API Gateway                               |
|                     (FastAPI + Uvicorn + Pydantic v2)                   |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                               Orchestrator                              |
|                 (Conversation Control & Multi-Turn Loop)                |
+-------------------------------------------------------------------------+
        |                                                 |
        v                                                 v
+-----------------------+                         +-----------------------+
|     Intent Router     |                         |     State Manager     |
| (Deterministic Rules  |                         |  (Session, Context,   |
|   + Model Classifier) |                         |  Execution Records)   |
+-----------------------+                         +-----------------------+
        |                                                 |
        v                                                 v
+-----------------------+                         +-----------------------+
| Model Interface Layer |                         |     Tool Executor     |
|  (OllamaAdapterABC)   |                         |  (Isolated Execution  |
|  + Tenacity Retries   |                         |  + Timeout & Schema)  |
+-----------------------+                         +-----------------------+
        |                                                 |
        v                                                 v
  [Local LLMs]                                      [Registered Tools]
  - Gemma 4 E4B (General)                           - Calculator
  - Qwen2.5-Coder 7B (Code)                         - System Diagnostic
  - Qwen3 8B (Reasoning)                            - Weather Service
  - DeepSeek-R1 8B (Complex)
```

---

## 3. Core Architectural Contracts (Abstract Base Classes)

To ensure module interchangeability and vendor neutrality, the system enforces the following ABCs in `app.models.base` and `app.core.abc`:

1. **`ModelConnectorABC`**:
   - `connect(api_key: Optional[str]) -> bool`: Handshake and availability check.
   - `generate(prompt: str, max_tokens: int) -> Optional[str]`: Unstructured text generation.
   - `generate_structured(messages: list[dict], tool_schemas: list[dict]) -> ModelResponse`: Function-calling generation returning tool requests or final text.
2. **`InputValidatorABC`**:
   - `validate(data: dict[str, Any]) -> ValidationReport`: Structural validation returning errors and recovery suggestions.
3. **`OutputParserABC`**:
   - `parse(raw_output: str) -> dict[str, Any]`: Text and JSON deserialization logic.
4. **`ToolExecutorABC`**:
   - `execute(tool_name: str, arguments: dict[str, Any]) -> ToolExecutionResult`: Safe execution with timeout.

---

## 4. Resilience and Failure Mitigation

1. **API Rate Limiting & Transient Failures**: Handled via Tenacity exponential backoff with randomized jitter (`wait_random_exponential`).
2. **Schema Drift (LLM Hallucinations)**: Arguments are validated with Pydantic. Any `ValidationError` is serialized into structured error feedback, injected into context, and given to the model for self-correction without crashing.
3. **Circular Loops**: Monitored via a strictly incremented iteration counter (`max_depth = 5`). Exceeding this boundary produces a controlled graceful conclusion.
4. **State Persistence Failures**: State updates are committed synchronously/transactionally before and after tool calls.

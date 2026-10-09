# Data Flow Specification: DFD Level 0 & Level 1

## DFD Level 0: System Context Diagram

```
[ User / Client ]
      |
      | 1. Submit Prompt, Session ID, Context
      v
+=============================================+
|                                             |
|   0. AI Model-Routing & Tool-Use System     |<===> [ SQLite Database ]
|                                             |      (Sessions, History, Logs)
+=============================================+
      |                                   |
      | 2. Inference & Schemas            | 3. Execute Tool Name & Args
      v                                   v
[ Local LLM Runtime (Ollama) ]    [ Registered Python Tools ]
```

---

## DFD Level 1: Decomposed Process Data Flow

### Process Breakdown:
1. **1.0 Receive & Validate**:
   - Ingress validation of HTTP JSON payload using `PromptRequest` schema.
   - Rejection of malformed requests with `422 Unprocessable Entity` or structured `ValidationReport`.
2. **2.0 Load & Manage State**:
   - Fetches historical conversation entries and last tool call from SQLite state store.
   - Constructs an active `AgentState` instance.
3. **3.0 Route & Invoke Model**:
   - **Tier 1 (Deterministic)**: Tests input against regex / rule patterns. If a match occurs, routes directly.
   - **Tier 2 (Intent Classification)**: Selects target model (e.g. `gemma-4-e4b`, `qwen2.5-coder:7b`, `qwen3:8b`, `deepseek-r1:8b`).
   - Dispatches formatted messages and tool schemas to the model adapter.
4. **4.0 Validate & Execute Tool**:
   - Inspects model response: if `tool_call`, verifies the tool is present in the `ToolRegistry`.
   - Validates arguments using Pydantic.
   - Executes with timeout. Writes `ToolExecution` entry to database.
   - Feeds tool result back to Process 3.0 context (loop iteration incremented).
5. **5.0 Synthesize & Validate**:
   - When the model returns terminal natural text, validates the output.
6. **6.0 Persist & Respond**:
   - Commits the turn to SQLite.
   - Returns `ChatResponse` containing final output, session state, route chosen, and tool execution traces.

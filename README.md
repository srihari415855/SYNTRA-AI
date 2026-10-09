# 🏗️ AI Model-Routing & Tool-Use System MVP

A high-performance, resilient tool-use harness and multi-turn model router built with **FastAPI**, **Pydantic v2**, **Tenacity**, **SQLAlchemy**, and **Ollama**.

## 🌟 Key Features
- **Multi-Turn Orchestrator Loop**: ReAct-style agentic loop with strict call-depth protection (`max_depth = 5`) preventing circular dependency locks.
- **Two-Tier Intent Routing**:
  - *Tier 1 (Fast Deterministic)*: Rule and regex matcher executing tools or specialized responses without model overhead.
  - *Tier 2 (Model-Based Classification)*: Intelligent dispatch between Primary (`Gemma 4 E4B`) and Specialists (`Qwen2.5-Coder 7B`, `Qwen3 8B`, `DeepSeek-R1 8B`).
- **Resilient Model Interface Layer (MIL)**: Native Ollama adapter with exponential backoff + jitter retries using Tenacity, plus graceful mock/fallback mode.
- **Sandboxed Tool Runtime**: Dynamic tool registry, automatic JSON Schema generation, Pydantic argument validation, bounded execution timeouts, and controlled error recovery.
- **Transactional Persistence**: SQLite + SQLAlchemy recording complete sessions, messages, and tool execution audit trails.
- **Structured JSON Logging**: Machine-readable observability tracking every request lifecycle.

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.11+
- [Ollama](https://ollama.com/) (Optional: Mock fallback is enabled if Ollama is not currently running)

### 2. Setup
```bash
# Navigate to directory
cd ai-system

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -e .
```

### 3. Run the API Server
```bash
python -m app.main
```
The server will start at `http://localhost:8000`. Interactive Swagger UI is available at `http://localhost:8000/docs`.

### 4. Running the Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py"
# Or run our test runner:
python tests/run_tests.py
```

---

## 📡 API Endpoints

- `POST /api/v1/chat`: Dispatch prompt through the orchestrator.
- `GET /api/v1/sessions/{session_id}`: Retrieve full session history and tool traces.
- `GET /api/v1/tools`: List all registered tools and their schemas.
- `GET /health`: System diagnostic health status.

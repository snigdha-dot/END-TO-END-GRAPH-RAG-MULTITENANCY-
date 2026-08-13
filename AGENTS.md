# AI Agent Guidelines (AGENTS.md)

## 🤖 Instructions for AI Coding Assistants
If you are an AI coding assistant (VS Code Copilot, AGY, Cursor, Claude Dev, etc.) working on this repository, you MUST strictly adhere to the following rules:

---

## 🛑 MANDATORY RULES

### 1. Zero Cross-Tenant Data Leakage
- NEVER bypass tenant database routing.
- EVERY database query must be targeted strictly to `/api/v1/command/{tenant_db}/cypher`.
- Validate `tenant_id` at the API boundary using `app/api/dependencies.py`.

### 2. Strict Cypher & SQL Parameterization (Anti-Injection)
- NEVER concatenate raw user strings into Cypher queries (e.g. `f"MATCH (n {name: '{user_input}'})"` IS BANNED).
- ALWAYS use parameter maps (`{"query": "MATCH (n {name: $name}) RETURN n", "params": {"name": user_input}}`).
- Enforce pre-approved schema enums for vertex labels and relationship edge types.

### 3. Fail-Safe Defensive Retrieval
- Retrieval queries MUST NEVER throw unhandled exceptions when a graph search returns zero nodes or when an entity is missing.
- If graph traversal returns empty, gracefully fall back to dense vector KNN chunk retrieval.

### 4. Mandatory Side-by-Side Telemetry Instrumentation
- Every new LLM call or retrieval step must record:
  1. Execution duration in milliseconds (`latency_ms`).
  2. Input/Output token counts and cost calculation in USD (`cost_usd`).
- Always attach telemetry metrics to the returned response payload or test runner output.

### 5. Documenting Changes in `documentation.md`
- Whenever you modify code, create a component, or run tests, add a timestamped entry to `documentation.md` following the established markdown log format.

---

## 🛠️ Code Style & Conventions
- Use Python 3.11+ type hints (`str`, `dict[str, Any]`, `list[Node]`).
- Use Pydantic v2 schemas for all API payloads.
- Use `async/await` for HTTP calls and I/O bound operations.
- Handle exceptions using custom error classes in `app/core/exceptions.py`.

# Project Development Log & Audit (documentation.md)

All engineering decisions, code changes, schema definitions, and benchmark test results for Team B are logged here with timestamps.

---

## 📅 Log Entries

### [2026-08-13 15:02:00 IST] - Repository & Architecture Initialization
* **Author**: Team B Lead Architect (AI Pair Developer)
* **Action**: Initialized Team B Multi-Tenant Graph RAG Service repository structure.
* **Key Artifacts Created**:
  * `team_b_master_plan.md`: Master architectural blueprint including Security, Chunking, Entity Disambiguation, Dual-Path Retrieval, and Side-by-Side Cost Telemetry.
  * `context.md`: Comprehensive project context document for AI agents and human developers.
  * `AGENTS.md`: Operational guidelines and security constraints for coding agents working on this codebase.
  * `documentation.md`: Chronological development audit log.
* **Security & Multi-Tenancy Decisions**:
  * Selected ArcadeDB Community Edition (Apache 2.0) with database-per-tenant isolation (`tenant_hr_kb`, `tenant_tech_kb`).
  * Enforced parameterized Cypher query generation to eliminate injection risks.
* **Telemetry & Cost Tracking Decisions**:
  * Implemented side-by-side reporting for step-level latency (ms) and model token/USD cost ($).

### [2026-08-13 15:03:30 IST] - Git Repository Setup & Initial Commit
* **Author**: Team B Lead Architect (AI Pair Developer)
* **Action**: Initialized Git repository and staged initial project files.
* **Details**:
  * Configured `.gitignore` targeting Python build files, virtual environments (`.venv`), IDE settings, SQLite databases, and ArcadeDB local data caches.
  * Executed `git init` in `scratch/team_b_graph_rag`.
  * Staged and committed core project documents (`.gitignore`, `context.md`, `AGENTS.md`, `documentation.md`).
  * Repository is ready for remote tracking (`git remote add origin <URL>`).

### [2026-08-13 15:07:30 IST] - Team B Core Codebase Implementation Complete
* **Author**: Team B Lead Architect (AI Pair Developer)
* **Action**: Built full production-ready FastAPI application & ArcadeDB multi-tenant client.
* **Components Built**:
  * `app/core/`: `config.py`, `security.py` (anti-Cypher injection & tenant verification), `telemetry.py` (side-by-side ms & $ cost tracker), `exceptions.py`.
  * `app/models/`: Pydantic V2 schemas for Graph primitives (`Vertex`, `Edge`, `Subgraph`), API payloads (`RetrievalRequest/Response`, `IngestionRequest/Response`), and Tenant Schema (`TenantSchemaConfig`).
  * `app/services/`:
    * `arcadedb_client.py`: Async connection pool for ArcadeDB REST API & Cypher execution.
    * `chunking_service.py`: Structure-aware semantic markdown chunker with parent-child hierarchy.
    * `extraction_service.py`: Hybrid NER + Rule relation extractor.
    * `resolution_service.py`: Entity Disambiguation algorithm via Jaro-Winkler string similarity ($\ge 0.85$) and canonical node merging.
    * `retrieval_service.py`: Dual-path hybrid search (ArcadeDB HNSW vector KNN + multi-hop Cypher traversal + RRF reranker + defensive zero-result fallback).
  * `app/api/v1/`: `retrieval.py` (`POST /api/v1/retrieval/search`), `ingestion.py` (`POST /api/v1/ingestion/document`), `tenant.py` (`POST /api/v1/tenant/create`, `GET /api/v1/tenant/schema`).
  * `app/main.py`: Main FastAPI entry point with CORS, OpenAPI spec, and health check routes.
  * `docker-compose.yml` & `Dockerfile`: Container configuration for ArcadeDB Community Edition & Team B API.
  * `tests/`: Comprehensive test suite including unit tests, Cypher injection security tests, tenant isolation tests, and end-to-end telemetry benchmarks (`test_pipeline_benchmark.py`).

### [2026-08-13] - Local Development Environment Provisioning
* **Author**: AI Pair Developer
* **Action**: Provisioned the local toolchain required to actually run the service. Prior log entries recorded code authored, not an environment stood up.
* **Installed**:
  * Python 3.11.9 (via `winget install Python.Python.3.11`) — the system default was 3.14.6, which lacks wheels for the pinned ML dependencies.
  * `.venv` created against 3.11.9.
  * Core runtime + test dependencies: fastapi 0.141.1, uvicorn 0.52.2, pydantic 2.13.4, pydantic-settings 2.15.0, httpx 0.28.1, numpy 2.4.6, rapidfuzz 3.14.5, pytest 9.1.1, pytest-asyncio 1.4.0, locust 2.46.3.
  * Docker Desktop 4.86.0 (via `winget install Docker.DockerDesktop`); Docker CLI 29.7.2 confirmed on PATH.
* **Verification Results**:
  * `pytest -q` → **7 passed**.
  * `GET /health` → 200, `{"status":"online","arcadedb_ready":false}`.
  * `POST /api/v1/retrieval/search` → 200 with full side-by-side telemetry payload; exercised the defensive zero-node fallback path as designed (ArcadeDB not yet running).
* **Deliberately Deferred**:
  * `torch`, `spacy`, `sentence-transformers`, `gliner` (~2.5GB) — declared in `requirements.txt` but not imported by any current code path. Installing them is not required to run or test the service today; needed only once real embeddings/NER are implemented.
* **Blocked — Requires Administrator Elevation + Reboot**:
  * WSL2 is **not installed** on this machine, so Docker Desktop cannot start its backend and ArcadeDB cannot be launched.
  * `wsl --install --no-distribution` fails from a non-elevated shell; enabling the `Microsoft-Windows-Subsystem-Linux` and `VirtualMachinePlatform` optional features requires admin rights.
  * Port 8000 is occupied by an unrelated user Python process (PID 1324); local verification used port 8010 instead.

### [2026-08-13] - Production Hardening Rebuild
* **Author**: AI Pair Developer
* **Action**: Rebuilt the service to production standard. Closed the fail-open defect, implemented all four security layers from the master plan, and built the retrieval intelligence that was previously scaffolded but absent.

#### Critical defect closed (P0)
`arcadedb_client` converted every exception into `[]`, and retrieval converted `[]` into a hardcoded fallback string. The service returned **HTTP 200 with plausible telemetry whether or not anything worked** — a caller could not distinguish a healthy system from a fully broken one. Failures now raise a typed error (`DatabaseConnectionError` 503 / `DatabaseQueryError` 502); only a query that executed and matched nothing returns an empty list. Verified live: with ArcadeDB down, a retrieval request now returns `503 database_unavailable`, where it previously returned `200` with fabricated context.

#### Security (master plan section 2.1, all four layers)
* **Layer 1** — `app/api/dependencies.py`. API keys now bind to exactly one tenant (`API_KEY_TENANT_MAP`); the credential determines the knowledge base. `X-Tenant-ID` is an assertion that is *verified*, not an override — a mismatch is 403. Previously any valid key could read any tenant by changing one header. Optional HS256 JWT layer verifies a signed `tenant_id` claim, cross-checked against the key's tenant; rejects `alg:none`, expired, and wrong-secret tokens. Constant-time key comparison.
* **Layer 2** — `app/core/tenant_context.py`. Async `contextvar` tenant guard; a database call with no bound context fails closed rather than running unscoped.
* **Layer 3** — `app/core/security.py`. Parameterizer now schema-driven: only identifiers validated against the tenant's approved schema AND matching a strict identifier pattern are ever interpolated. Injection guard covers statement chaining, comments, procedure calls, unions, template interpolation, null bytes, and oversized input. Depth clamped to 3, limit to 100.
* **Layer 4** — database-per-tenant routing, with auto-provisioning disabled by default (it was an unbounded disk-exhaustion vector and masked tenant typos as empty knowledge bases).
* **Middleware** — `app/api/middleware.py`: per-credential rate limiting, structured audit logging (key fingerprints, never keys or content), request IDs, body-size limits, and security headers.
* **Secrets** — all moved to `.env`; `validate_production()` refuses to start a production process holding placeholder secrets, wildcard CORS, or autoprovisioning.

#### Retrieval intelligence (master plan section 4)
* **Entity linking rewritten.** Seeds were fabricated from raw query tokens — "Which films did the director of Inception make?" produced `canon_which`, `canon_films`, `canon_did`. Those match nothing, so the graph path could never return a result even against a fully populated database. Now: stopword-filtered mention extraction -> parameterized candidate lookup -> vector + Jaro-Winkler confirmation against real canonical nodes.
* **Path A (vector KNN) built** — `embedding_service.py` (bge-small-en-v1.5) plus HNSW index DDL in `graph_schema_service.py`. The `arcadedb_vector_knn` telemetry key required by plan section 5 is now always emitted; it was previously absent from every response, silently breaking the documented contract.
* **`reranker_service.py` created** — RRF fusion (rank-based, so incomparable vector and graph scores can be combined), cross-encoder reranking, and degree-centrality subgraph pruning.
* **Defensive fallback made real** — a genuine broader vector sweep instead of a synthetic sentence.
* **Per-tenant relationship vocabularies** — `tenant_schema.py`. The hardcoded `DEPENDS_ON/OWNS/MANAGES` list was infrastructure vocabulary, useless for real domains. Movies now uses `DIRECTED/ACTED_IN/PRODUCED_BY`; AI trends uses `BUILDS_ON/RELEASED_BY/TRAINED_ON`.
* **Extraction** — GLiNER (zero-shot, labels drawn from tenant schema) with spaCy and regex fallbacks; LLM provider interface defined and stubbed per FOSS-only scope.
* **Chunking to spec** — 100-token overlap (previously none, so facts spanning a boundary were lost), real tokenizer via tiktoken (was `len//4`), markdown section hierarchy and sibling links.
* **Resolution** — blocking keys replace the O(n^2) all-pairs comparison; embedding-assisted matching catches semantic aliases below the string threshold; canonical ids are label-scoped so "Dune" the film and "Dune" the studio stay distinct.
* **Ingestion batched** — one HTTP round-trip per vertex *and* per edge is replaced by batched writes.

#### Testing
* **140 tests passing**, 7 integration tests gated on a live ArcadeDB.
* **Real cross-tenant leakage tests** (plan section 7 row 1) — the previous file only regex-checked a header and never attempted cross-tenant access.
* JWT forgery, `alg:none` downgrade, injection, traversal bounds, rate limiting, fail-loud behaviour, telemetry contract, RRF properties, and multi-hop entity linking all covered.
* Locust suite now runs a continuous isolation probe under load.

#### Live verification (ArcadeDB intentionally down)
| Probe | Result |
| :--- | :--- |
| Missing / invalid API key | 401 |
| Cross-tenant access attempt | **403 tenant_access_denied** |
| Cypher injection payload | **400 security_violation** |
| Unknown request field | 422 |
| Legitimate query, database down | **503 database_unavailable** (was a fake 200) |
| Security headers + request IDs | present |

#### Deliberately deferred
* ML dependencies (`torch`, `sentence-transformers`, `gliner`, `spacy`) split into `requirements-ml.txt`. The service runs and is fully testable without them, degrading to lexical embeddings, lexical reranking, and regex NER. `/ready` reports which mode is active so degraded quality is never silent.
* A concrete LLM provider: the interface exists, `NullLLMProvider` is active, matching the agreed FOSS-only scope.
* Rate limiting is in-process; a multi-instance deployment should move it to Redis.

#### Artifacts
`INTEGRATION.md` (Team A contract), `.env.example`, `requirements-ml.txt`, hardened multi-stage `Dockerfile` (non-root user, readiness healthcheck), `docker-compose.yml`.

#### Known blocker
WSL2 is not installed, so Docker Desktop cannot start and ArcadeDB has never run. The 7 integration tests and the end-to-end multi-hop proof remain unexecuted until then. Run `wsl --install --no-distribution` as Administrator, reboot, then `docker compose up -d arcadedb`.



### [2026-08-14 IST] - Production Evaluation Harness (Retrieval Quality + Tenant Isolation)
* **Author**: Team B Lead Architect (AI Pair Developer)
* **Action**: Built a labelled evaluation harness that produces auditable production metrics rather than assertions. Commit `ff1e6b7`.

#### Artifacts created
| File | Purpose |
| :--- | :--- |
| `tests/evaluation/dataset.py` | Two deliberately disjoint tenant corpora (`movies_bot`, `ai_trends_bot`) with 22 labelled questions carrying ground-truth entity ids; 8 multi-hop, plus edge cases and cross-domain negatives; 8 adversarial injection payloads |
| `tests/evaluation/metrics.py` | Recall@k, Precision@k, MRR, nDCG@k implemented directly so figures are auditable; entity-linking rate, fallback rate, error rate, latency percentiles; `multi_hop_advantage()` for graph-vs-vector lift |
| `tests/evaluation/isolation_suite.py` | Bidirectional leakage with real data, direct entity-id probing, 50 interleaved concurrent cross-tenant queries, tenant-id edge cases (path traversal, unicode homoglyph, null byte, injection), fail-closed guard verification |
| `tests/evaluation/run_evaluation.py` | Provisions, ingests, scores, runs the isolation battery, emits the report |
| `tests/evaluation/report.py` | Markdown + JSON with explicit hard/soft gates |

#### Design decisions
* **Disjoint corpora by construction.** The two tenant vocabularies do not overlap, so any cross-tenant hit is unambiguous evidence of a leak rather than coincidental term overlap.
* **Isolation is the only hard commercial gate.** Retrieval quality is a tuning problem; cross-tenant leakage invalidates the product regardless of quality scores.
* **Ablation control added.** `execute_retrieval()` gains `disable_graph_path`, reducing the pipeline to vector-only so the graph path's actual contribution can be isolated. `retrieval_diagnostics` gains `graph_path_disabled`.
* **The runner refuses to emit metrics without a live ArcadeDB** (exit code 2). Numbers produced by the fallback path would describe the harness rather than the system — the same fail-open failure mode this service was hardened against in the previous entry.

#### Verified (offline paths only)
* 12/12 isolation checks pass: tenant-id validation across 11 hostile inputs, plus the unscoped-access fail-closed guard.
* Preflight correctly refuses and exits 2 when ArcadeDB is unreachable.

#### Not yet executed
Every quality metric (Recall@k, MRR, nDCG@k, graph lift) and the live half of the isolation battery (leakage, entity probing, concurrency, injection). These require a running database.

#### Blocker unchanged
`wsl --install --no-distribution` requires Administrator; the agent session runs unelevated and cannot raise a UAC prompt. Docker Desktop cannot start without WSL2, and Java is absent so the non-Docker ArcadeDB path is also unavailable. All three routes to a live database converge on an elevated installer plus a reboot.

**To produce the report:**
1. `wsl --install --no-distribution` (Administrator PowerShell), then reboot
2. Launch Docker Desktop; `docker compose up -d arcadedb`
3. `.\.venv\Scripts\python.exe -m tests.evaluation.run_evaluation`
4. Output: `reports/EVALUATION_REPORT.md` (+ timestamped `.md`/`.json`)

Optional but recommended: `pip install -r requirements-ml.txt` (~2.5GB, no admin needed) before running. Without it the harness runs on lexical fallbacks and quality metrics understate real performance; the report flags this at the top. Isolation results are unaffected either way.

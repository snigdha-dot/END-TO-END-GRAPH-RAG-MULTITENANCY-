# Team B Master Plan: Multi-Tenant Graph RAG Engine

## Executive Focus
**Team B Scope**: Design, build, secure, and maintain the **Shared Multi-Tenant Graph RAG Retrieval Service & Knowledge Base Ingestion Pipeline** powered by ArcadeDB.

---

## 📄 1. Mandatory Project Knowledge & Agent Files

To ensure seamless collaboration across human developers and AI coding agents (VS Code, AGY, Cursor, etc.), Team B maintains three core persistent documentation files in the project root:

1. **`context.md`**: Complete architectural context, team boundaries (Team A vs Team B), multi-tenant ArcadeDB routing model, data models, and API specifications. Any new agent reading this file gains instant full comprehension of the project.
2. **`AGENTS.md`**: Explicit operational rules, coding standards, security constraints (zero raw Cypher concatenation, tenant parameter isolation), and testing requirements for AI agents modifying this repository.
3. **`documentation.md`**: Chronological timestamped log of all development decisions, schema changes, feature additions, bug fixes, and system benchmark results.

---

## 🔒 2. Security & Multi-Tenancy Architecture

Security and isolation are zero-compromise requirements. Team B's engine must guarantee that no tenant can read, modify, or leak data from another tenant.

### 2.1 Tenant Isolation & Security Layers
```
Incoming Team A Request
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Security Layer 1: API Authentication Middleware       │
│  - Validates API Key / JWT token                      │
│  - Verifies token claims against requested tenant_id   │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│  Security Layer 2: Tenant Context Guard                │
│  - Sets thread-local / async context contextvar        │
│  - Strictly restricts DB connection pool scope         │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│  Security Layer 3: Query Parameterizer                 │
│  - NO string concatenation for Cypher/SQL queries      │
│  - Parameterized queries with target DB scope          │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│  Security Layer 4: ArcadeDB Database-Level Isolation   │
│  - Query targets: `/api/v1/command/{tenant_db}/cypher` │
└────────────────────────────────────────────────────────┘
```

### 2.2 Defense Against Cypher Injection & Attacks
* **Strict Parameterization**: Every user query parameter is bound to query parameters (`$entity_id`, `$vector_embedding`, `$max_hops`). Never concatenate raw text into Cypher strings.
* **AST & Schema Whitelisting**: Relationship types and vertex labels in traversal queries are strictly validated against the target tenant's pre-approved schema enum before query construction.
* **Query Execution Limits**:
  * Mandatory query timeouts (`timeout: 3000ms`).
  * Upper bounds on traversal depth (`max_depth: 3`) and max result nodes (`limit: 100`) to prevent graph traversal explosions or Denial of Service (DoS).

---

## 📥 3. High-Precision Ingestion Pipeline (Chunking & Graph Construction)

A Graph RAG is only as good as its ingestion pipeline. To pass rigorous testing, we employ a multi-stage chunking, extraction, and resolution workflow.

```
Raw Documents (PDF / Markdown / HTML / TXT)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Step 1: Semantic & Contextual Chunking                 │
│ - Structure-aware chunking (Markdown headers / paras)  │
│ - Chunk size: 400 - 600 tokens with 100 token overlap  │
│ - Parent-Child hierarchy (Child chunks linked to Parent│
│   document context)                                    │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ Step 2: High-Precision Hybrid Entity Extraction         │
│ - Fast NER (GLiNER / spaCy) for candidate entities     │
│ - LLM JSON-schema extraction for domain-specific       │
│   entities & explicit relationships                    │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ Step 3: Entity Disambiguation & Resolution Algorithm   │
│ - Candidate lookup via HNSW vector similarity          │
│ - String similarity score (Jaro-Winkler >= 0.85)       │
│ - LLM verification for ambiguous entities              │
│ - Canonical Entity Node merge in ArcadeDB              │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ Step 4: Relationship Filtering & Schema Validation     │
│ - Edge confidence thresholding (Confidence >= 0.80)   │
│ - Schema constraint check (SourceType -> Edge -> Target│
│ - Deduplication & provenance link creation             │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
 ArcadeDB Graph Ingestion (Vertices, Edges, HNSW Embeddings)
```

---

## 🔎 4. Multi-Hop Hybrid Retrieval & Telemetry

To pass testing scenarios where vector-only RAG fails, Team B implements a 4-stage hybrid retrieval algorithm:

```
Team A Query: "What components are impacted if Database X fails?"
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Query Intent Parsing & Entity Linking         │
│ - Extract seed entities from query ("Database X")      │
│ - Generate dense embedding of query (bge-small-en-v1.5)│
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Dual-Path Hybrid Search                       │
│ Path A: ArcadeDB Vector Index KNN search for top-K     │
│         relevant text chunks                           │
│ Path B: Seed Entity identification + Multi-Hop Cypher  │
│         graph traversal (1 to 3 hops)                  │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Subgraph Pruning & Reranking                  │
│ - Merge Vector Chunks + Graph Subgraph                 │
│ - Reciprocal Rank Fusion (RRF) algorithm               │
│ - Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)     │
│ - Filter irrelevant nodes based on centrality          │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ Stage 4: Defensive Fallback & Telemetry Instrumentation│
│ - If Graph traversal returns 0 results -> Fallback to  │
│   pure dense vector search automatically (Zero Fail)   │
│ - Compute side-by-side latency & model token cost      │
│ - Format structured JSON for Team A                    │
└────────────────────────────────────────────────────────┘
```

---

## 📊 5. Side-by-Side Latency & Price Cost Telemetry Specification

Every retrieval response and test run output includes detailed side-by-side performance and cost metrics to ensure total transparency over operational expenses and performance bottlenecks.

### Response Telemetry Format:
```json
{
  "tenant_id": "chatbot_tech_support",
  "query": "What components are impacted if Database X fails?",
  "subgraph": { "nodes": [], "edges": [] },
  "context_passages": [],
  "telemetry": {
    "latency_breakdown_ms": {
      "query_entity_linking": 14.2,
      "arcadedb_vector_knn": 18.5,
      "arcadedb_cypher_traversal": 22.1,
      "rrf_reranking": 9.4,
      "total_retrieval_latency": 64.2
    },
    "model_cost_breakdown": {
      "models_called": [
        {
          "step": "query_intent_parsing",
          "model_name": "bge-small-en-v1.5",
          "tokens_used": { "prompt_tokens": 18, "completion_tokens": 0, "total_tokens": 18 },
          "cost_usd": 0.000000,
          "latency_ms": 14.2
        },
        {
          "step": "entity_disambiguation_llm",
          "model_name": "gemini-1.5-flash",
          "tokens_used": { "prompt_tokens": 142, "completion_tokens": 12, "total_tokens": 154 },
          "cost_usd": 0.000015,
          "latency_ms": 185.0
        }
      ],
      "total_request_cost_usd": 0.000015
    }
  }
}
```

---

## 🏗️ 6. Project Codebase & Documentation Layout

```
scratch/team_b_graph_rag/
├── context.md                       # Comprehensive architecture & project context for all AI agents
├── AGENTS.md                        # Strict agent coding rules & security guidelines
├── documentation.md                 # Timestamped log of changes, tests, and milestones
├── app/
│   ├── api/
│   │   ├── dependencies.py          # Auth & Tenant context validation
│   │   ├── middleware.py            # Security, Rate Limiting, Audit logs
│   │   └── v1/
│   │       ├── retrieval.py         # POST /api/v1/retrieval/search
│   │       ├── ingestion.py         # POST /api/v1/ingest/document
│   │       └── schema.py            # GET/POST /api/v1/tenant/schema
│   ├── core/
│   │   ├── config.py                # Environment & Security configs
│   │   ├── security.py              # Token validation & Cypher parameterizer
│   │   ├── telemetry.py             # Latency & Cost tracking engine
│   │   └── exceptions.py            # Custom error handlers
│   ├── services/
│   │   ├── arcadedb_client.py       # Async HTTP client pool for ArcadeDB
│   │   ├── chunking_service.py      # Semantic & Contextual chunker
│   │   ├── extraction_service.py    # Hybrid NER + LLM relationship extractor
│   │   ├── resolution_service.py    # Entity disambiguation & merging
│   │   ├── retrieval_service.py     # Hybrid Search & Cypher traversal
│   │   └── reranker_service.py      # RRF & Cross-Encoder reranker
│   └── models/
│       ├── tenant.py                # Tenant metadata & DB mapping
│       ├── payload.py               # Request / Response Pydantic schemas
│       └── graph.py                 # Vertex, Edge, Triple dataclasses
├── tests/
│   ├── unit/                        # Chunker, Entity resolver, Cypher builder tests
│   ├── integration/                 # ArcadeDB container integration tests
│   ├── security/                    # Multi-tenant data leakage & Cypher injection tests
│   └── evaluation/                  # Latency & cost side-by-side benchmark harness
├── docker-compose.yml               # Local ArcadeDB + Team B API setup
├── Dockerfile                       # Production container build
└── requirements.txt                 # Open-source dependencies
```

---

## 🧪 7. Testing & Quality Assurance Suite

| Test Category | Description | Success Criterion |
| :--- | :--- | :--- |
| **Multi-Tenant Leakage Test** | Query Tenant A's endpoint attempting to retrieve known entity IDs in Tenant B. | Must return `404` or empty result; zero cross-tenant leakage. |
| **Cypher Injection Test** | Pass malicious Cypher fragments (`' RETURN count(*) MATCH (n) --`) into user query parameters. | Query parameterizer neutralizes injection cleanly. |
| **Empty Graph Fallback Test** | Query an un-indexed entity or isolated node in the graph. | System automatically falls back to vector semantic search without error. |
| **Side-by-Side Cost Telemetry** | Execute pipeline test harness. | Outputs side-by-side latency (ms) and price ($) breakdown per step and model. |

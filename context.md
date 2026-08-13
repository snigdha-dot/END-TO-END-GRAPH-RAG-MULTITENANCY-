# Project Context: Team B Multi-Tenant Graph RAG Service

## 📌 Executive Summary & Purpose
This codebase contains the **Team B Multi-Tenant Graph RAG Retrieval Service** powered by ArcadeDB.

The system acts as a high-performance **Retrieval-as-a-Service (RaaS)** platform. Multiple domain-specific chatbots (owned by Team A) send retrieval requests to this service. The service executes security-guaranteed tenant isolation, entity resolution, vector KNN search, and multi-hop Cypher graph traversal over ArcadeDB, returning structured subgraphs and text context.

---

## 🏛️ System Architecture & Team Boundaries

```
[ Team A: Chatbot Services ] (UI, Auth, Redis Memory, LLM Generation)
            │
            │ HTTP POST /api/v1/retrieval/search
            │ Payload: { tenant_id: "xyz", query: "..." }
            ▼
[ Team B: Retrieval Service Engine ] (FastAPI Python 3.11+)
            │
            ├─► Entity Resolver & Disambiguator (GLiNER / spaCy / Local Embeddings)
            ├─► Security Guard & Cypher Parameterizer (Anti-Injection & Tenant Scope)
            ├─► RRF Reranker & Fallback Engine (Vector + Graph Subgraph)
            └─► Telemetry Engine (Side-by-side Latency ms + Model Cost $ tracking)
            │
            ▼
[ ArcadeDB Multi-Tenant Server ] (Apache 2.0 Graph DB)
            ├── DB: `tenant_hr_kb`
            ├── DB: `tenant_tech_kb`
            └── DB: `tenant_legal_kb`
```

### Team Division of Responsibilities
* **Team A**: Owns the Chatbot UI, authentication, per-user conversation memory (Redis/PostgreSQL), domain system prompts, and final LLM response generation.
* **Team B (THIS REPOSITORY)**: Owns document chunking, entity extraction, entity resolution, graph schema validation, ArcadeDB multi-tenancy routing, graph traversal, vector search, RRF reranking, security, and side-by-side latency/cost telemetry.

---

## 🔒 Security & Multi-Tenancy Rules
1. **Tenant Isolation**: Every incoming request contains a `tenant_id`. The database router MUST map this to `/api/v1/command/{tenant_db}/cypher`. Zero data leakage across tenants is permitted.
2. **Cypher Parameterization**: Never use Python string formatting (`f"MATCH (n) WHERE n.name = '{input}'"`) to construct Cypher queries. All inputs MUST be parameterized.
3. **Traversal Bounds**: All Cypher queries must specify `max_depth` ($\le 3$) and `limit` ($\le 100$) to prevent traversal explosions.

---

## 📥 Ingestion & Retrieval Specifications
* **Chunking**: Semantic markdown/paragraph chunking ($400\text{--}600$ tokens, 100 token overlap) with Parent-Child context links.
* **Entity Disambiguation**: Vector candidate search $\rightarrow$ Jaro-Winkler string match ($\ge 0.85$) $\rightarrow$ Canonical vertex merging in ArcadeDB.
* **Hybrid Retrieval**: Dense Vector KNN (ArcadeDB HNSW) + Multi-Hop Cypher Traversal + Reciprocal Rank Fusion (RRF) Reranking.
* **Defensive Fallback**: If graph traversal returns zero nodes, the system automatically falls back to dense vector chunk retrieval.

---

## 📊 Telemetry & Side-by-Side Metrics
Every retrieval call and test run reports:
1. **Latency Breakdown (ms)**: `query_entity_linking`, `arcadedb_vector_knn`, `arcadedb_cypher_traversal`, `rrf_reranking`, `total_retrieval_latency`.
2. **Model Cost Breakdown ($)**: Per LLM/embedding model call token counts and USD price calculation side-by-side.

---

## 🛠️ Tech Stack & Dependencies
* **Language**: Python 3.11+
* **Framework**: FastAPI + Uvicorn (Async HTTP)
* **Graph Database**: ArcadeDB Community Edition (Apache 2.0)
* **Entity Extraction**: `gliner` / `spacy`
* **Embeddings**: `sentence-transformers` (`bge-small-en-v1.5` / `all-MiniLM-L6-v2`)
* **Vector Indexing**: Native ArcadeDB HNSW Vector Indexing

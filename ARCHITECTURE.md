# Architecture — Team B Multi-Tenant Graph RAG Retrieval Service

**Status:** running end-to-end against a live ArcadeDB. Last verified 2026-08-15.
**Scope:** everything from a raw document to the passages returned to a chatbot,
plus what has and has not been measured.

Every number in this document was measured on this machine. Where something is
unverified or degraded, it says so.

---

## 1. What this system is

A shared **Retrieval-as-a-Service** layer. Multiple chatbots (Team A) send natural
language queries; this service returns facts — a subgraph, text passages, and
telemetry. It does **not** generate prose and holds **no** conversation state.

```
[ Team A: chatbots ]  UI · user auth · Redis/Postgres memory · LLM generation
         │  HTTPS POST /api/v1/retrieval/search   X-API-Key: <per-chatbot>
         ▼
[ Team B: THIS SERVICE ]  FastAPI · Python 3.11.9
         ├─ Security: key→tenant binding, JWT, contextvar guard, parameterizer
         ├─ Ingestion: chunk → embed → extract → resolve → write
         ├─ Retrieval: link → dual-path → RRF → rerank → fallback
         └─ Telemetry: per-step latency (ms) + per-model cost (USD)
         ▼
[ ArcadeDB 24.11.1 ]  one physical database per tenant
         ├─ tenant_movies_bot_kb
         └─ tenant_ai_trends_bot_kb
```

**The division of responsibility is the contract:** Team B returns facts, Team A
turns facts into prose.

---

## 2. Multi-tenancy

### 2.1 Isolation model

**Database-per-tenant, not row-level.** The tenant's database name is part of the
request URL, so another tenant's data is not addressable — there is no
`WHERE tenant_id = ...` filter that could be forgotten.

```
X-API-Key: mv_live_…  →  tenant "movies_bot"  →  tenant_movies_bot_kb
                          POST /api/v1/command/tenant_movies_bot_kb
```

Implemented in `app/services/arcadedb_client.py::resolve_database`.

### 2.2 Four security layers

| Layer | Implementation | File |
| :--- | :--- | :--- |
| 1. Authentication | API key → exactly one tenant, constant-time compare (`hmac.compare_digest`). Optional HS256 JWT whose signed `tenant_id` claim is cross-checked against the key. | `app/api/dependencies.py`, `app/core/security.py` |
| 2. Tenant context guard | `contextvars.ContextVar` bound per request. Per-asyncio-task, so concurrent requests cannot observe each other. Unscoped DB access **raises** rather than running. | `app/core/tenant_context.py` |
| 3. Query parameterization | All values bound as parameters. Only schema-validated identifiers are ever interpolated, and re-checked against a strict pattern immediately before use. Depth ≤3, limit ≤100. | `app/core/security.py` |
| 4. Database isolation | DB name in the URL path. | `app/services/arcadedb_client.py` |

**The critical property:** a supplied `X-Tenant-ID` header is an assertion to be
*verified*, never an instruction to obey. A mismatch returns 403 and is logged as a
cross-tenant attempt. The earlier implementation validated only that a key existed
in a set, so any valid key could read any tenant by changing one header.

### 2.3 Per-tenant graph schemas

Each tenant declares its own vocabulary (`app/core/tenant_schema.py`). This is what
makes traversal useful: a single shared edge list (`DEPENDS_ON`, `OWNS`, `MANAGES`)
is infrastructure vocabulary and matches neither domain.

| Tenant | Vertex labels | Edge types |
| :--- | :--- | :--- |
| `movies_bot` | Film, Person, Studio, Genre, Award, Character, Country, Chunk | DIRECTED, ACTED_IN, WROTE, PRODUCED, COMPOSED_FOR, HAS_GENRE, PRODUCED_BY, WON_AWARD, NOMINATED_FOR, SEQUEL_OF, PLAYED_CHARACTER, RELEASED_IN, MENTIONED_IN |
| `ai_trends_bot` | Model, Organization, Technique, Paper, Person, Dataset, Benchmark, Hardware, Chunk | RELEASED_BY, BUILDS_ON, AUTHORED, USES_TECHNIQUE, TRAINED_ON, EVALUATED_ON, OUTPERFORMS, CITES, SUPERSEDES, RUNS_ON, AFFILIATED_WITH, MENTIONED_IN |

**Known limitation:** these are hardcoded in Python. Adding a tenant with a new
domain (e.g. an Ayurveda bot needing `Herb`/`Dosha`/`TREATS`) requires editing two
files and redeploying. Schemas belong in per-tenant config files. See §9.

### 2.4 Data currently in the system

Measured directly against the running server:

| Database | Chunks | Entities (by type) |
| :--- | ---: | :--- |
| `tenant_movies_bot_kb` | 116 | Person 773, Film 244 |
| `tenant_ai_trends_bot_kb` | 99 | Person 156, Model 83, Organization 16 |

Source: English Wikipedia article extracts, fetched via the public REST API
(`tests/evaluation/corpus_loader.py`). Two deliberately **disjoint** domains, so any
cross-tenant hit is unambiguous evidence of a leak rather than vocabulary overlap.

---

## 3. Ingestion pipeline

`POST /api/v1/ingest/document` → `app/services/ingestion_service.py`

```
raw document
   → chunk           (structure-aware, overlapping)
   → embed           (384-dim vectors)
   → extract         (entities + typed relations)
   → resolve         (canonical merge)
   → validate        (schema + confidence)
   → write           (concurrent, parameterized)
```

### 3.1 Chunking — `chunking_service.py`

| Aspect | Choice | Why |
| :--- | :--- | :--- |
| Tokenizer | `tiktoken` (`cl100k_base`), word-count fallback | `len(text)//4` drifts badly on real prose |
| Structure | Markdown heading hierarchy via regex | Section breadcrumb lets a chunk saying "It grossed $836M" become answerable |
| Size | 400–600 tokens, target 500 | Master plan spec |
| Overlap | 100 tokens, on a word boundary | Without it a fact spanning a boundary is lost from both chunks |
| Hierarchy | `parent_doc_id` + `prev_chunk_id`/`next_chunk_id` + `section_path` | Parent-child context linking |

Oversized paragraphs split on sentence boundaries. Chunks never cross a section
boundary.

### 3.2 Embeddings — `embedding_service.py`

| Aspect | Choice |
| :--- | :--- |
| Model | `bge-small-en-v1.5` via `sentence-transformers` 5.7.0 |
| Dimensions | 384, L2-normalized |
| Query prefix | `"Represent this sentence for searching relevant passages: "` — BGE is trained asymmetrically; omitting it measurably hurts recall |
| Loading | Lazy, on first use (import costs seconds and hundreds of MB) |
| Fallback | Deterministic SHA1 bag-of-words hashing into 384 dims |

The fallback is not semantic — it captures lexical overlap only. It exists so the
pipeline stays exercisable without model weights, and `is_semantic` reports which
mode ran so telemetry never claims quality it did not deliver.

**Current state: running on the fallback.** `torch` 2.13.0+cpu is installed but
fails to load (`c10.dll` missing its dependencies) because the Microsoft VC++
Redistributable is absent and installing it needs Administrator.

### 3.3 Entity extraction — `extraction_service.py`, `type_inference.py`

Three backends in preference order:

1. **GLiNER** (`urchade/gliner_small-v2.1`) — zero-shot: entity labels are supplied
   at inference time from the tenant's own schema, so movies extracts
   `Film/Person/Studio` and AI-trends extracts `Model/Organization/Technique` with
   no retraining. **Not currently active** (blocked with torch).
2. **spaCy** (`en_core_web_sm`) — fixed label set mapped onto the tenant schema.
3. **Regex + contextual type inference** — the active backend.

The regex path was rewritten because a shared default label broke everything
downstream. `TypeInferencer` resolves a label from context, ordered by reliability:

| Signal | Example | Confidence |
| :--- | :--- | ---: |
| Copular definition | "Inception **is a** 2010 **film**" | 0.85 |
| Document subject | opening sentence types the article's subject | 0.82 |
| Verb cue | "**directed by** Christopher Nolan" → Person | 0.72 |
| Name shape | corporate suffix → Studio; `GPT-4` pattern → Model | 0.65 |
| Schema default | — | 0.45 |

Duplicate mentions reconcile to the highest-confidence reading, so a passing
reference does not override a definitional one.

Noise filtering matters on real prose: sentence openers ("However", "Although"),
dates and month names, citation furniture ("ISBN", "Retrieved"), and section
headings are rejected. Without it the graph fills with `The` and `January`.

### 3.4 Relation extraction — `relation_patterns.py`

The first version matched only active-voice `X verb Y` and produced **3 edges from
417,000 characters**. Encyclopedic prose is mostly not shaped that way.

| Construction | Example | Handling |
| :--- | :--- | :--- |
| Passive + agentive `by` | "written and directed **by** Christopher Nolan" | Inverts direction: yields `Nolan DIRECTED Inception` |
| Coordinated objects | "directed Interstellar, Dunkirk **and** Oppenheimer" | Emits an edge to each, with a confidence decay per position |
| Active | "Hans Zimmer composed the score for Dune" | Direct |
| Sentence bounds | — | Pairs never cross a sentence boundary |

Each rule declares the endpoint labels it permits (`Person DIRECTED Film` is valid;
`Person DIRECTED Person` is rejected), so type inference and relation extraction
reinforce each other. Confidence threshold ≥0.80.

### 3.5 Entity resolution — `resolution_service.py`

| Step | Technique |
| :--- | :--- |
| Normalization | lowercase, punctuation stripped, spaces → underscores |
| Blocking | 3-char prefix + token keys, **label-scoped** — avoids O(n²) comparison |
| String match | Jaro-Winkler ≥0.85 via `rapidfuzz` 3.14.5 — prefix-weighted, right for names |
| Vector rescue | Cosine ≥0.80 when the string score is 0.60–0.85, catching "OpenAI" vs "Open AI Inc" (JW ≈0.83, below threshold) |
| Canonical ID | `canon_{label}_{normalized}` — **label-scoped**, so the film "Dune" and the studio "Dune" stay distinct |
| Merge | aliases collected, mention counts summed, highest-confidence surface form becomes the display name |

### 3.6 Graph write — `arcadedb_client.py`

All writes are `MERGE` (idempotent). Values are bound parameters; only
schema-validated labels are interpolated.

Two performance findings, both discovered only under real data volume:

**Untyped endpoint matching was a cartesian product.**
`MATCH (a {entity_id: …}), (b {entity_id: …})` scans every vertex type for both
endpoints. Naming both labels uses the UNIQUE index instead:
**65,316ms → 1,144ms (57×).**

**Serial writes dominated ingestion.** ArcadeDB 24.11.1 exposes no working batch
endpoint (`/api/v1/batch/{db}` returns 404), and `sqlscript` does not accept bound
parameters — using it would trade the injection guarantee for throughput, so it was
rejected. Instead: bounded concurrency (semaphore of 6) with one backoff-and-retry,
treating 503 as back-pressure rather than a dead database.
**395s → 66.5s per 52K-char document (~6×).** 16 concurrent writers made the
single-node server return 503; 6 is the measured sustainable bound.

Entities are also capped per chunk (25, by confidence), but relations are extracted
*first* and their endpoints always retained, so nothing participating in an edge is
lost. 25% fewer writes.

---

## 4. Storage layer

**ArcadeDB 24.11.1 Community Edition** (Apache 2.0), running standalone on a
portable JDK 21 — no Docker, no WSL2, no Administrator rights.

| Aspect | Detail |
| :--- | :--- |
| Access | REST API over `httpx` 0.28.1 async pool (50 connections, 20 keepalive) |
| Query languages | cypher, sql, sqlscript, gremlin, graphql, mongo, js, java |
| Timeouts | reads 3000ms (plan bound), writes 30s, DDL 60s |
| Indexes | UNIQUE on `entity_id`, NOTUNIQUE on `normalized_name`, `parent_doc_id` |

**Vector index:** HNSW is specified but **not available** — ArcadeDB 24.11.1 does
not expose vector index creation through SQL DDL. Vector search falls back to
in-process cosine scoring over a bounded candidate set. Correct results, worse
asymptotics; fine at current scale, will not scale. The probe result is cached per
tenant so a failing query is not retried on every search.

### 4.1 ArcadeDB compatibility findings

Six behaviours that differ from the Cypher/Neo4j assumptions the code was written
against. All were invisible until the database actually ran:

| Finding | Consequence |
| :--- | :--- |
| `/api/v1/ready` returns **204**, not 200 | A healthy server was reported unavailable |
| `label` is a reserved TinkerPop token | Cannot be set as a vertex property; renamed `entity_label` |
| `ANY(x IN list WHERE …)` unsupported | Alias matching moved into the resolution service |
| `nodes(path)` / `relationships(path)` unimplemented | Traversal projects endpoints, then recovers typed edges in a second bounded query |
| No batch endpoint | Concurrent sequential writes instead |
| Cypher engine ~8.5s first-call warmup, then ~120ms | Timeouts split by operation class |

---

## 5. Retrieval pipeline

`POST /api/v1/retrieval/search` → `app/services/retrieval_service.py`

### Stage 1 — Query intent parsing & entity linking

```
query → stopword-filtered mention extraction → DB lookup → vector + string confirm → seed IDs
```

Mentions are extracted preferring capitalized spans, with leading interrogatives
stripped ("Which Films" → "Films") and a lowercase content-word fallback.

**This stage was previously fabricated.** Seeds came from raw query tokens, so
*"Which films did the director of Inception make?"* produced `canon_which`,
`canon_films`, `canon_did` — stopwords matching nothing. The graph path could never
return a result even against a fully populated database.

### Stage 2 — Dual-path hybrid search

| Path | Method |
| :--- | :--- |
| **A: Vector KNN** | Query embedded with the BGE prefix; cosine over chunk embeddings. Native HNSW when available, in-process scoring otherwise. |
| **B: Graph traversal** | Bounded multi-hop Cypher from the linked seeds. |

```cypher
MATCH (start)-[rel:DIRECTED|:ACTED_IN|:PRODUCED_BY*1..2]-(related)
WHERE start.entity_id IN $start_nodes
RETURN start.entity_id AS source_id, start.name AS source_name,
       start.entity_label AS source_label,
       related.entity_id AS target_id, related.name AS target_name,
       related.entity_label AS target_label
LIMIT $limit
```

Seeds and limit are bound parameters. Depth clamped ≤3, limit ≤100. Relationship
types come from the tenant's schema and are re-validated before interpolation.

**This is where multi-hop answers come from.** For *"which other films did the
director of Inception make?"*, vector search retrieves the Inception page and stops
— no chunk states the connection. Traversal walks
`Inception → Nolan → {Interstellar, Dunkirk}` in two hops.

Traversed entities are bridged back to text via `MENTIONED_IN`.

### Stage 3 — Fusion, reranking, pruning — `reranker_service.py`

| Step | Technique |
| :--- | :--- |
| **RRF** | `Σ 1/(k + rank)`, k=60. Fuses by **rank**, not score, because cosine similarity and hop-distance are not comparable scales. Items found by both paths rank highest. |
| **Cross-encoder** | `ms-marco-MiniLM-L-6-v2`, reads query+passage jointly. Bounded to the top slice (O(n) model calls). Lexical-overlap fallback when unavailable. |
| **Centrality pruning** | Degree centrality, seeds always retained |

### Stage 4 — Defensive fallback & telemetry

If both paths return nothing, a **broader pure-vector sweep** runs. It previously
emitted a hardcoded string, which meant the service returned HTTP 200 with
plausible telemetry whether or not anything worked.

If the database is unreachable it now raises **503**, not a fabricated 200.

### Response shape

```json
{ "tenant_id": "...", "subgraph": {"nodes": [...], "edges": [...]},
  "context_passages": ["Christopher Nolan (Person) directed Inception (Film)."],
  "linked_entities": [...],
  "telemetry": {
    "latency_breakdown_ms": {"query_entity_linking": …, "arcadedb_vector_knn": …,
      "arcadedb_cypher_traversal": …, "rrf_reranking": …, "total_retrieval_latency": …},
    "model_cost_breakdown": {"total_request_cost_usd": 0.0},
    "retrieval_diagnostics": {"seed_entity_count": …, "graph_nodes": …,
      "fallback_used": false, "semantic_embeddings": …, "native_hnsw_knn": …}
  },
  "request_id": "..." }
```

Graph-derived passages come **first** — multi-hop answers live in edges, not in any
single chunk.

---

## 6. Generation — deliberately absent

**This service never calls an LLM.** Team A owns generation: they inject
`context_passages` into their prompt alongside their system prompt and conversation
memory, then call their own model.

The LLM interface for *extraction* and *disambiguation* exists
(`LLMExtractionProvider`) with `NullLLMProvider` active, matching the agreed
FOSS-only scope. Pricing for `gemini-1.5-flash`, `gemini-1.5-pro`, and `gpt-4o-mini`
is configured so the cost telemetry works the moment a provider is wired in. Wiring
one is a config change, not a refactor.

All active models are local and free, so `total_request_cost_usd` is `0.0`.

---

## 7. Security controls

| Control | Implementation |
| :--- | :--- |
| Authentication | API key → single tenant, constant-time compare |
| Tokens | In-tree HS256 JWT: rejects `alg:none`, verifies `exp`/`nbf`/`iss`/`aud`, cross-checks `tenant_id` against the key. **Currently disabled** (`JWT_ENABLED=false`) |
| Injection | Parameter binding primary; pattern guard as defence-in-depth (statement chaining, schema destruction, procedure calls, comments, template interpolation, null bytes) |
| Tenant IDs | Validated against `^[a-z][a-z0-9_]{1,62}$` — **rejected, never normalized**, so a typo cannot silently route elsewhere |
| Traversal bounds | depth ≤3, limit ≤100, 3000ms timeout |
| Rate limiting | Sliding window per key fingerprint. **In-process** — with N workers the effective limit is N× configured |
| Audit logging | Key fingerprint only, never the key, query text, or retrieved content |
| Headers | `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, CSP, HSTS in production |
| Secrets | `.env`, with boot-time production validation that refuses placeholder values |
| Provisioning | Explicit admin action; auto-provisioning on query is disabled (it was a disk-exhaustion vector) |

---

## 8. Evaluation — what has actually been measured

### 8.1 Harness

| Component | Purpose |
| :--- | :--- |
| `dataset.py` | 22 synthetic labelled questions + 8 adversarial payload classes |
| `corpus_loader.py` | Real corpora: Wikipedia REST (no credentials), Kaggle CSV, or local files. Cached, with exponential backoff |
| `real_questions.py` | 31 ground-truth questions verified against fetched article text |
| `metrics.py` | Recall@k, Precision@k, MRR, nDCG@k implemented directly so figures are auditable; entity-linking rate, fallback rate, latency percentiles, `multi_hop_advantage()` |
| `isolation_suite.py` | The multi-tenancy battery |
| `run_evaluation.py` | Orchestration, vector-only ablation, preflight guard |

The runner **refuses to emit metrics without a live database** (exit 2). Numbers
from the fallback path would describe the harness rather than the system — the same
fail-open failure this service was hardened against.

### 8.2 Multi-tenancy results — **37/37 PASS**

| Category | Checks | Result |
| :--- | ---: | :--- |
| Cross-tenant leakage (bidirectional, real corpora) | 8 | PASS |
| Entity-ID probing (asking tenant B for A's exact IDs) | 8 | PASS |
| Injection defence (8 payload classes) | 8 | PASS |
| Tenant-ID validation (path traversal, unicode homoglyph, null byte, SQL) | 11 | PASS |
| Concurrent interleaved load (50 cross-tenant requests) | 1 | PASS |
| Unscoped access fails closed | 1 | PASS |

**Zero foreign entities returned in either direction.** The sharpest case: *"Who
directed Inception?"* asked against `ai_trends_bot` returned nothing and correctly
triggered the fallback — the boundary held for a perfect match on the *other*
tenant's content.

### 8.3 Retrieval quality — latest run (4 docs/tenant, real Wikipedia)

| Metric | Before fixes | After fixes |
| :--- | ---: | ---: |
| Pass rate | 100% | 100% |
| **Graph lift** | **−0.75** | **+0.30** |
| Vector-only hit rate | 1.00 | 0.70 |
| Hybrid hit rate | 0.25 | 1.00 |
| Questions only the graph solved | 0 | **3 of 10** |
| Fallback rate | — | 0% |
| p95 latency | 351 / 268 ms | 3585 / 1163 ms |

**Graph lift is the number that justifies the architecture.** 3 of 10 multi-hop
questions were answered by traversal and could not be answered by vector-only
search. Every question ran with `fallback_used=false` — real retrieval throughout.

**Caveat on Recall@5 and MRR:** the run reports `recall@5=1.000` with `MRR=0.000`.
This is a scoring artifact, not a perfect score. Real-corpus questions are scored on
`expected_text` because entity IDs vary by NER backend, so `expected_entities` is
empty — `recall_at_k` returns 1.0 vacuously when there is nothing to find, while
`reciprocal_rank` returns 0.0. **The trustworthy metrics from this run are pass
rate, graph lift, and fallback rate.**

### 8.4 Extraction improvement

| Corpus | Edges before | Edges after |
| :--- | ---: | ---: |
| movies (417K chars) | 3 | 173 |
| ai_trends (636K chars) | 2 | 56 |

Entity types went from 1 (everything `Person`) to 6–8 correctly distinguished types.

### 8.5 Performance

| Operation | Measured |
| :--- | ---: |
| Ingestion, 52K-char document | 66.5 s (was 395 s) |
| Ingestion, 4 documents | 176 s (movies) / 66 s (ai_trends) |
| Edge write | 1.1 s (was 65.3 s) |
| Retrieval p95 | 1.2–3.6 s |

### 8.6 Test suite

**140 tests passing** (7 integration tests deselected without a database): unit
tests for chunking, resolution, RRF; security tests for cross-tenant leakage, JWT
forgery, `alg:none` downgrade, injection, traversal bounds, rate limiting;
fail-loud behaviour; telemetry contract.

---

## 9. Known limitations

**Running degraded.** `torch` cannot load (VC++ Redistributable missing, needs
Administrator), so embeddings are lexical hashing, reranking is lexical overlap, and
NER is the regex heuristic. `/ready` reports the active mode, so degradation is
never silent — but quality metrics understate real performance, and `Person` remains
over-represented in extraction.

**Schemas are hardcoded.** New domains require editing Python and redeploying.
They belong in per-tenant config files. Worth fixing *before* onboarding a third
tenant, since canonical IDs are label-scoped and a schema change means re-ingesting.

**HNSW unavailable** in ArcadeDB 24.11.1 via SQL DDL; vector search is in-process
cosine over a bounded candidate set.

**Ingestion is slow** at ~44 s/document even after the 6× improvement.

**p95 latency of 3.6 s** is well above the 500 ms gate.

**Production infrastructure not built:** TLS, load balancer, multiple pods, Redis
(rate limiting and caching), separate ingestion workers, metrics/alerting. Model
inference is synchronous CPU work on the event loop — while it runs, that worker
serves nobody, capping throughput regardless of the async design. Details in
`PRODUCTION_READINESS.md`.

**Untested:** sustained load (Locust suite written, never run), and behaviour at
realistic corpus scale — the largest test so far is 215 chunks across two tenants.

---

## 10. Technology summary

| Layer | Technology | Version |
| :--- | :--- | :--- |
| Language | Python | 3.11.9 |
| API | FastAPI + Uvicorn | 0.141.1 / 0.52.2 |
| Validation | Pydantic v2 | 2.13.4 |
| HTTP client | httpx (async) | 0.28.1 |
| Graph DB | ArcadeDB CE (Apache 2.0) | 24.11.1 |
| Runtime | Temurin JDK (portable) | 21.0.12 |
| Embeddings | sentence-transformers / `bge-small-en-v1.5` | 5.7.0 |
| Reranking | `ms-marco-MiniLM-L-6-v2` cross-encoder | — |
| NER | GLiNER / spaCy / regex+type-inference | — |
| String similarity | RapidFuzz (Jaro-Winkler) | 3.14.5 |
| Tokenization | tiktoken (`cl100k_base`) | 0.13.0 |
| Testing | pytest / pytest-asyncio / Locust | 9.1.1 / 1.4.0 / 2.46.3 |

All components are Apache-2.0 or MIT and run locally. Total model cost: **$0.00**.

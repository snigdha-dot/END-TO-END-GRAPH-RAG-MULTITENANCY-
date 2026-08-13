# Production Readiness — Outstanding Work

**Date:** 2026-08-13
**Service:** Team B Multi-Tenant Graph RAG Retrieval Service
**Status:** Application code complete and tested (140 tests passing). Infrastructure and operational verification outstanding.

---

## 1. Summary

Everything **inside** the API process is built and verified. Everything **around** it is not.

| Layer | State |
| :--- | :--- |
| Auth chain, tenant isolation, injection defence | Built, verified live |
| Graph RAG pipeline (link → dual-path → RRF → rerank) | Built, unit-tested |
| Ingestion (chunk → embed → extract → resolve → batch write) | Built, unit-tested |
| Telemetry (latency ms + cost USD) | Built, verified |
| TLS, load balancer, multiple pods, Redis, ingestion workers | **Not built** |
| End-to-end verification against a real database | **Never executed** |

The single most misleading property right now: horizontal scaling *looks* free because the service is stateless, but the in-process rate limiter means adding pods silently multiplies the effective limit. Redis is not optional beyond one pod.

---

## 2. BLOCKING — cannot serve external traffic without these

### 2.1 TLS / HTTPS
**Status:** Does not exist.
**Risk:** API keys travel as plaintext headers (`X-API-Key: mv_live_…`). Anyone on the network path can read and reuse them to query a tenant's knowledge base.
**Fix:** Terminate TLS at a load balancer or reverse proxy (nginx, HAProxy, AWS ALB, Cloudflare, Kubernetes Ingress). Application code is unchanged — the app keeps serving plain HTTP on 8000 behind it.
**Owner:** Infrastructure.

### 2.2 Event-loop blocking on model inference
**Status:** Application defect. Code exists but blocks.
**Risk:** `embedding_service.encode()` and the cross-encoder are synchronous CPU work called directly inside async handlers. While inference runs, **that worker serves no other request**. Estimated ceiling ~50 req/s per pod regardless of async design — this silently negates the concurrency architecture.
**Fix:** `run_in_executor` with a bounded thread pool in `app/services/embedding_service.py` and `app/services/reranker_service.py`.
**Owner:** Application. **No infrastructure required — can be done immediately.**

### 2.3 Redis-backed rate limiting
**Status:** In-process only (`app/api/middleware.py` holds counters in a Python dict).
**Risk:** Each worker and each pod keeps its own window. Configured 120 req/min becomes 120 × workers × pods. Counters reset on every deploy. Contains a buggy client looping; does **not** stop a determined attacker.
**Fix:** Move the sliding window to Redis, keyed by API key fingerprint.
**Owner:** Application (client) + Infrastructure (Redis instance).

### 2.4 Enable JWT
**Status:** Fully implemented and tested; disabled via `JWT_ENABLED=false`.
**Risk:** Static API keys never expire, cannot be revoked without a coordinated redeploy on both sides, and carry no user identity — so an audit log cannot attribute a request to a user or detect Team A leaking one user's context into another's.
**Fix:** Set `JWT_ENABLED=true`, agree the signing secret with Team A, coordinate their token issuance.
**Owner:** Application config + Team A coordination.

### 2.5 Durable audit logging
**Status:** Structured logs to stdout only.
**Risk:** No retention, no query capability, no alerting. Likely insufficient for compliance and useless for post-incident forensics.
**Fix:** Ship to a durable sink (CloudWatch, Loki, ELK, Datadog) with retention policy and alerts on `tenant_access_denied` and `security_violation`.
**Owner:** Infrastructure.

---

## 3. NEVER EXECUTED — verification gaps

### 3.1 Integration tests (highest priority after ArcadeDB is up)
**Status:** 7 tests written in `tests/integration/`, never run — they require a live ArcadeDB.

Includes `test_cross_tenant_data_is_not_reachable`, which ingests a distinctive entity into one tenant and asserts it is unreachable from another. **This is the only test that proves the core isolation guarantee with real data.**

Current isolation evidence: verified at the auth layer (403 on cross-tenant attempts, confirmed live) and by construction (database name is in the URL path). **Not** verified by experiment.

Also unproven until these run: that the ingestion Cypher is accepted by a real ArcadeDB, that HNSW index creation succeeds on this build, and that a genuine multi-hop traversal returns real data.

```
wsl --install --no-distribution      # as Administrator, then reboot
docker compose up -d arcadedb
pytest tests/integration -v
```

### 3.2 Load testing
**Status:** Locust suite written (`tests/evaluation/locustfile.py`), never run.
**Consequence:** Throughput is entirely unmeasured. The ~50 req/s figure in 2.2 is architectural reasoning, not measurement. Deployment sizing is currently guesswork.
The suite includes a continuous cross-tenant isolation probe that asserts 403 under load.

### 3.3 ArcadeDB has never run
No document has ever been ingested into a real database; no query has ever returned real retrieved data. Blocked on WSL2 install + reboot.

---

## 4. NOT BUILT — infrastructure

| Component | State | Note |
| :--- | :--- | :--- |
| **Load balancer** | Does not exist | Wire `/ready` (not `/health`) as the health check so a pod with a broken DB connection is pulled from rotation |
| **Multiple API pods** | Single container only | Service is stateless, so this is straightforward — but only after 2.3 |
| **Redis** | Does not exist | Needed for rate limiting (2.3) and response caching |
| **Separate ingestion workers** | Does not exist | Ingestion currently shares the retrieval connection pool; a bulk ingest degrades live queries. Should be queue-driven on its own pool |
| **Metrics / alerting** | None | Per-request telemetry exists in responses, but no aggregate view. Prometheus / OpenTelemetry |
| **Response cache** | None | Chatbot queries repeat heavily; per-tenant keyed cache is a large latency win |

---

## 5. OPERATIONAL

- **Real API keys** — current values are `dev_movies_key_change_me` placeholders. Generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Note: `INTEGRATION.md` promises 403 on key/tenant mismatch, which is implemented — but do not send that document to Team A with placeholder keys.
- **ML dependencies** — `requirements-ml.txt` (~2.5GB) not installed. The service runs and is fully testable without them but degrades to lexical hashing embeddings, lexical reranking, and regex NER. Semantic quality is materially lower. `/ready` reports the active mode, so degradation is never silent.
- **Dependency scanning** — no `pip-audit` or Dependabot in CI.
- **External security review** — nobody adversarial has examined this.
- **Team A inputs still outstanding** — chatbot identifiers, source corpus, **10–20 sample questions per bot including multi-hop ones** (highest-value: drives per-tenant relationship vocabularies and enables quality measurement), traffic estimates, staging contact.

---

## 6. Recommended order

1. `wsl --install --no-distribution` (Administrator) → reboot → `docker compose up -d arcadedb`
2. **Run integration tests** — prove isolation with real data before anything else
3. **Run Locust** — get a real throughput number instead of an estimate
4. **Fix event-loop blocking** (2.2) — application code, no infrastructure, likely the real bottleneck
5. **Redis + enable JWT** (2.3, 2.4)
6. **TLS + load balancer + multiple pods** (2.1)
7. Real keys, ML deps, durable audit sink (2.5), metrics
8. Separate ingestion workers, response cache
9. Dependency scanning, external security review

Items 4 and 5 are application code and can proceed in parallel with 1–3.

---

## 7. What is already done

For contrast, verified live against the running API with ArcadeDB intentionally down:

| Probe | Result |
| :--- | :--- |
| Missing / invalid API key | 401 |
| Cross-tenant access attempt | **403 `tenant_access_denied`** |
| Cypher injection payload | **400 `security_violation`** |
| Unknown request field | 422 |
| Legitimate query, database down | **503 `database_unavailable`** |
| Security headers + request IDs | Present |

Plus: 140 tests passing, all four master-plan security layers implemented, dual-path retrieval with real entity linking, RRF fusion, per-tenant relationship vocabularies, batched idempotent ingestion, and the side-by-side latency/cost telemetry contract.

**The critical defect closed:** the service previously returned HTTP 200 with plausible telemetry whether or not anything worked — the database client swallowed every exception into an empty result and retrieval turned that into a hardcoded passage. A caller could not distinguish a healthy system from a dead one.

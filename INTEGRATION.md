# Team A Integration Guide

Shared multi-tenant Graph RAG retrieval service. One endpoint serves every chatbot;
each chatbot gets its own API key, and **the key determines which knowledge base is
queried**. You cannot reach another chatbot's data, by design.

---

## 1. Credentials

You receive one API key per chatbot, delivered separately over a secure channel.

| Chatbot | Tenant | Key |
| :--- | :--- | :--- |
| Movies | `movies_bot` | `mv_live_…` |
| AI Trends | `ai_trends_bot` | `ai_live_…` |

Store them as environment variables, never in source control. Each key is scoped to
exactly one knowledge base.

**Base URLs**
- Staging: `http://<staging-host>:8000`
- Production: TBC

Interactive OpenAPI docs are served at `<base-url>/docs` and always match the
deployed build.

---

## 2. Retrieval

`POST /api/v1/retrieval/search`

```http
POST /api/v1/retrieval/search
X-API-Key: <your chatbot's key>
Content-Type: application/json

{
  "user_query": "Which other films did the director of Inception make?",
  "options": {
    "max_traversal_depth": 2,
    "top_k": 5,
    "include_vector_search": true,
    "include_subgraph": true
  }
}
```

| Option | Range | Default | Meaning |
| :--- | :--- | :--- | :--- |
| `max_traversal_depth` | 1–3 | 2 | Graph hops. `2` answers "the director of X also made…" |
| `top_k` | 1–20 | 5 | Maximum passages returned |
| `include_vector_search` | bool | `true` | Disable to use graph traversal only |
| `include_subgraph` | bool | `true` | Set `false` to omit the graph and shrink the payload |

### Response

```json
{
  "tenant_id": "movies_bot",
  "query": "Which other films did the director of Inception make?",
  "subgraph": { "nodes": [...], "edges": [...] },
  "context_passages": [
    "Christopher Nolan (Person) directed Inception (Film).",
    "Christopher Nolan (Person) directed Interstellar (Film)."
  ],
  "chunks": [...],
  "linked_entities": [
    {"mention": "Inception", "entity_id": "canon_film_inception",
     "name": "Inception", "label": "Film", "score": 1.0, "method": "exact"}
  ],
  "telemetry": { "latency_breakdown_ms": {...}, "model_cost_breakdown": {...} },
  "request_id": "6f1c…"
}
```

**`context_passages` is the field you want** — inject it into your LLM prompt.
Graph-derived passages come first, because multi-hop answers live in relationships
rather than in any single chunk of text.

The other fields are optional: `subgraph` for a sources/provenance view,
`linked_entities` to see which entities we resolved (useful when debugging a poor
answer), `telemetry` for latency and cost attribution, `request_id` to quote in
support requests.

### Minimal client

```python
import httpx

async def get_context(query: str) -> list[str]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{BASE_URL}/api/v1/retrieval/search",
            headers={"X-API-Key": MOVIES_BOT_KEY},
            json={"user_query": query, "options": {"top_k": 5}},
        )
        if response.status_code != 200:
            return []                      # degrade, do not fail the user's turn
        return response.json()["context_passages"]
```

---

## 3. Ingestion

`POST /api/v1/ingest/document` — chunks, embeds, extracts entities, resolves
duplicates, and indexes into your tenant graph.

```json
{
  "doc_id": "inception_wiki",
  "content": "# Inception\n\nInception is a 2010 film directed by Christopher Nolan...",
  "metadata": {"source": "wikipedia", "updated": "2026-08-13"}
}
```

Idempotent: re-posting the same `doc_id` updates in place rather than duplicating.
Requires the `ingestion:write` scope when JWTs are enabled.

---

## 4. Errors

| Code | Meaning | What to do |
| :--- | :--- | :--- |
| `200` | Success | Use `context_passages` |
| `401` | Missing or invalid API key | Check configuration |
| `403` | Key not authorized for the requested tenant | You sent another chatbot's tenant |
| `404` | Knowledge base not provisioned | Contact Team B |
| `422` | Malformed request body | Check against the schema |
| `429` | Rate limit exceeded | Honour `Retry-After`; back off |
| `503` | Knowledge base unavailable | Retry with backoff |

Every error carries `{"error": "...", "detail": "...", "request_id": "..."}`.

**Always degrade gracefully.** On any non-200, answer from conversation history or
say you lack information — never fail the user's turn outright.

Default rate limit is **120 requests/minute per key** (plus a burst allowance).
Responses carry `X-RateLimit-Limit` and `X-RateLimit-Remaining`.

---

## 5. Optional: JWT for per-user traceability

API keys authenticate the *chatbot*. Adding a JWT identifies the *user*, which
gives us per-user audit trails, short-lived credentials, and scopes.

```http
X-API-Key: <chatbot key>
Authorization: Bearer <jwt>
```

Claims: `tenant_id` (required, must match your key's tenant), `user_id`, `exp`,
`iss`, `aud`, `scope` (`retrieval:read`, `ingestion:write`).

A token whose `tenant_id` disagrees with the API key is rejected with `403` — a
token cannot be replayed against a different chatbot's credential.

Coordinate the signing secret with Team B before enabling.

---

## 6. Division of responsibility

**Team B (this service)** — chunking, embeddings, entity extraction and resolution,
graph traversal, vector search, RRF fusion and reranking, tenant isolation,
latency/cost telemetry. **We return facts, not prose. We never call an LLM and hold
no conversation state.**

**Team A (you)** — UI, user authentication, conversation memory, domain system
prompts, and final answer generation.

---

## 7. What we need from you

1. **Chatbot list** — name and knowledge domain per bot, so we can provision the
   knowledge base and issue a key. Confirm `movies_bot` and `ai_trends_bot`, or send
   your preferred identifiers.
2. **Source documents** per chatbot, plus who owns updates and how often they change.
3. **10–20 representative user questions per chatbot**, including several multi-hop
   ones. **This is the highest-value thing you can send early** — it drives the
   per-tenant relationship vocabulary (`DIRECTED`/`ACTED_IN` for movies,
   `BUILDS_ON`/`RELEASED_BY` for AI trends) and lets us measure retrieval quality.
4. **Expected traffic** — peak requests/second and acceptable p95 latency.
5. **A staging contact** for integration testing.

### One contract note

`tenant_id` in the request body is **deprecated and ignored for routing** — the API
key alone determines the tenant. It is accepted for backward compatibility and will
be removed. Please do not depend on it.

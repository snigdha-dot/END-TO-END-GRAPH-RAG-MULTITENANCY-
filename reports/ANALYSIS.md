# Evaluation Analysis — First Live Run

**Date:** 2026-08-15
**Environment:** ArcadeDB 24.11.1 (standalone, portable JDK 21 — no Docker, no WSL2)
**Mode:** lexical fallback (ML dependencies unavailable — see §4)

---

## 1. Headline

| Area | Result |
| :--- | :--- |
| **Multi-tenancy isolation** | **PASS — 37/37 checks** |
| Injection defence | PASS — 8/8 payloads rejected |
| Pipeline end-to-end | **Working** — provision → ingest → link → traverse → retrieve, against a real database |
| Retrieval quality | **Poor — Recall@5 0.295 / 0.111.** Root cause identified (§3) |
| Graph lift over vector-only | **-0.75 (negative).** Root cause identified (§3) |

The isolation result is the commercially significant one and it is unambiguous. The
quality result is bad, but the cause is a single identified defect rather than an
architectural problem.

---

## 2. Multi-tenancy — verified with real data

This is the first time the isolation guarantee has been tested against real ingested
corpora rather than asserted by construction.

| Category | Checks | Result |
| :--- | ---: | :--- |
| Cross-tenant leakage (bidirectional, real corpora) | 8 | PASS |
| Entity-id probing (asking tenant B directly for A's ids) | 8 | PASS |
| Injection defence (8 payload classes) | 8 | PASS |
| Tenant-id validation (path traversal, unicode homoglyph, null byte, SQL) | 11 | PASS |
| Concurrent interleaved load (50 requests across tenants) | 1 | PASS |
| Unscoped access fails closed | 1 | PASS |

Two corpora with deliberately disjoint vocabularies were ingested into separate
databases. Every distinctive phrase and canonical entity id from each tenant was
queried against the other. **Zero foreign entities were returned in either
direction.** Under 50 interleaved concurrent cross-tenant requests, zero violations.

Notably, the negative-control question "Who directed Inception?" asked against
`ai_trends_bot` returned nothing and correctly triggered the defensive fallback —
the tenant boundary held even when the query was a perfect match for the other
tenant's content.

---

## 3. Why retrieval quality is poor — one root cause

**The regex extractor cannot distinguish entity types.** Every extracted entity was
labelled `Person`, including films and studios:

```
Christopher Nolan -> canon_person_christopher_nolan   (correct)
Inception         -> canon_person_inception           (should be canon_film_inception)
```

Canonical ids are label-scoped by design, so that the film "Dune" and the studio
"Dune" remain distinct entities. With every label collapsed to `Person`, the ids the
ground-truth set expects (`canon_film_inception`) never match what was written
(`canon_person_inception`). Recall is therefore measuring an id-mismatch, not a
retrieval failure — the traversal genuinely found Nolan → Inception, and the passage
`"Christopher Nolan (Person) directed Inception (Person)"` is correct apart from the
label.

This also explains the **negative graph lift (-0.75)**. The vector-only ablation
scores on text overlap and is unaffected by labels, so it scores 1.0 while the
graph path is penalised for mislabelled ids. The comparison is not currently
measuring what it is meant to measure.

**Secondary effects of the same cause:**
- Only 3 and 2 relationships extracted from corpora containing dozens. Relation
  patterns are gated on schema-valid edge types, and `Person DIRECTED Person` is not
  valid in the movies schema — so most edges were correctly rejected as
  schema violations.
- `ai_trends_bot` scored worse (0.111) because its entity names are acronyms and
  hyphenated model names (`GPT-4`, `HNSW`), which the capitalisation heuristic
  handles worse than proper nouns.

**Fix:** GLiNER zero-shot NER, which takes the tenant's own label set at inference
time and is precisely the component the master plan specifies for this. It is
installed by `requirements-ml.txt` and blocked only by the environment issue in §4.

---

## 4. Environment limitations encountered

**ML dependencies could not be installed.** `torch` requires the Microsoft Visual
C++ Redistributable, whose installer needs Administrator rights unavailable in this
session. Consequently:

| Component | Intended | Actual |
| :--- | :--- | :--- |
| Embeddings | `bge-small-en-v1.5` (semantic) | lexical hashing |
| Reranking | `ms-marco-MiniLM-L-6-v2` cross-encoder | lexical overlap |
| NER | GLiNER zero-shot | regex capitalisation heuristic |

The service degraded exactly as designed and reported its mode in `/ready` and in
the report header, so the degradation was never silent. But it means **quality
metrics understate real performance**, and the NER fallback is the direct cause of
§3.

**ArcadeDB 24.11.1 does not expose HNSW index creation via SQL DDL.** Vector search
falls back to in-process cosine scoring over a bounded candidate set. Correct
results, worse asymptotics — fine at this corpus size, would not scale.

---

## 5. Defects found and fixed by this run

Six defects were invisible until the pipeline ran against a real database. All are
fixed and committed (`452efe9`, `b6e6f4e`):

1. **Readiness probe** expected HTTP 200; ArcadeDB answers **204**, so a healthy
   server was reported unavailable.
2. **`label` is a reserved TinkerPop token** and cannot be a vertex property.
   Ingestion failed with HTTP 500. Renamed to `entity_label`.
3. **`ANY(x IN list WHERE ...)`** is rejected by ArcadeDB Cypher. Alias matching
   moved to the resolution service, which already scores aliases.
4. **`nodes(path)` / `relationships(path)` are not implemented.** Traversal now
   projects endpoints and recovers typed edges in a second bounded query.
   Variable-length matching itself works, so multi-hop is unaffected.
5. **`/api/v1/batch/{db}` is absent.** Probe-once-and-remember, then sequential
   writes. `sqlscript` was rejected as an alternative because it does not accept
   bound parameters, and parameter binding is a security requirement.
6. **Cypher engine has a ~8.5s first-call warmup** settling to ~120ms. Timeouts are
   now split by operation class: 3000ms reads (the plan's bound), 30s writes,
   60s DDL.

This is the value of running against a real system: every one of these would have
reached production as a 500-level failure.

---

## 6. Performance

| Metric | movies_bot | ai_trends_bot |
| :--- | ---: | ---: |
| p95 latency | 351 ms | 268 ms |
| Ingestion | 3 docs / 10 chunks / 12.8 s | 3 docs / 5 chunks / 21.0 s |

Ingestion is slow because writes are sequential (§5.5) and include the Cypher warmup.
Retrieval latency is acceptable but was measured with lexical embeddings; real
embedding inference will add to it, and that inference is currently synchronous on
the event loop — the throughput ceiling flagged in `PRODUCTION_READINESS.md`.

---

## 7. Verdict

**Multi-tenancy: proven.** 37/37 with real data, real queries, concurrent load, and
adversarial input. This is the guarantee the product depends on and it now has
evidence behind it rather than an argument.

**Graph RAG: functionally working, quality unproven.** The pipeline demonstrably
links entities, traverses the graph, and returns real subgraphs and passages from a
real database. The quality numbers are not yet a valid measurement of the
architecture, because the extractor feeding it cannot type entities.

**Next step, in order:**
1. Install the VC++ Redistributable (Administrator), then `requirements-ml.txt`.
2. Re-run: `python -m tests.evaluation.run_evaluation`.
3. Re-read §3 — with GLiNER supplying correct labels, Recall@5 and graph lift
   become meaningful for the first time.

Until step 3, treat the quality figures in `EVALUATION_REPORT.md` as a measurement
of the regex fallback, not of the Graph RAG design.

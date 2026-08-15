# Edge Case Evaluation Report


## Result: 98/100 queries passed

> **Degraded mode.** Embeddings are lexical hashing rather than `bge-small-en-v1.5`, and reranking is lexical overlap rather than a cross-encoder. Semantic categories understate real performance; security and routing results are unaffected.

| Metric | Value |
| :--- | ---: |
| Total queries | 100 |
| Passed | 98 |
| Failed | 2 |
| Pass rate | 98.0% |
| Errors | 0 |
| Fallback rate | 3.0% |
| **Security verdict** | **PASS** |

## Per-category results

| # | Category | What it tests | N | Pass | Rate | p50 | p95 |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | basic_semantic | Vector retrieval | 12 | 12 | 100% | 300ms | 5059ms |
| 2 | exact_entity | BM25 + entity resolution | 8 | 8 | 100% | 233ms | 1692ms |
| 3 | section_context | Chunking quality | 8 | 8 | 100% | 76ms | 1650ms |
| 4 | relationship | Graph retrieval | 12 | 12 | 100% | 416ms | 17494ms |
| 5 | multi_hop | Graph traversal | 10 | 10 | 100% | 756ms | 5512ms |
| 6 | comparison | Multi-source retrieval + fusion | 8 | 8 | 100% | 89ms | 1388ms |
| 7 | global_community | Community search | 8 | 8 | 100% | 74ms | 134ms |
| 8 | structured_data | CSV/JSON/XLSX retrieval | 8 | 8 | 100% | 512ms | 1246ms |
| 9 | cross_document | Multi-document retrieval | 6 | 6 | 100% | 189ms | 2174ms |
| 10 | ambiguous | Query understanding | 5 | 3 | 60% | 63ms | 177ms |
| 11 | no_answer | Abstention | 5 | 5 | 100% | 184ms | 191ms |
| 12 | adversarial | Robustness | 5 | 5 | 100% | 0ms | 0ms |
| 13 | multi_tenant_isolation | Security | 5 | 5 | 100% | 65ms | 240ms |

## Latency

| Percentile | Milliseconds |
| :--- | ---: |
| p50 | 196.1 |
| p95 | 2700.7 |
| p99 | 5512.2 |
| mean | 733.5 |
| max | 17493.9 |

### Where the time goes

| Stage | Mean | p95 | Share |
| :--- | ---: | ---: | ---: |
| arcadedb_vector_knn | 612.7ms | 2564.12ms | 74.5% |
| graph_expansion | 163.3ms | 336.46ms | 19.8% |
| query_entity_linking | 15.75ms | 51.17ms | 1.9% |
| defensive_vector_fallback | 13.61ms | 13.85ms | 1.7% |
| community_search | 11.98ms | 62.17ms | 1.5% |
| context_optimization | 4.09ms | 5.75ms | 0.5% |
| rrf_reranking | 1.32ms | 2.51ms | 0.2% |
| rank_fusion | 0.08ms | 0.14ms | 0.0% |
| query_understanding | 0.07ms | 0.12ms | 0.0% |
| arcadedb_cypher_traversal | 0.0ms | 0.0ms | 0.0% |
| lexical_search | 0.0ms | 0.0ms | 0.0% |

## Cost

| Metric | USD |
| :--- | ---: |
| Total for this run | $0.000000 |
| Mean per query | $0.00000000 |
| Projected per 1,000 queries | $0.0000 |

All models run locally under FOSS licences and no LLM is called, so cost is a measured zero rather than an unmeasured one. The pricing matrix is configured, so these figures become non-zero the moment a provider is enabled.

## Failures

| Category | Query | Why |
| :--- | :--- | :--- |
| ambiguous | `more` | no passages returned |
| ambiguous | `herbs` | no passages returned |

## Environment

| Setting | Value |
| :--- | :--- |
| semantic_embeddings | False |
| embedding_model | lexical-fallback |
| cross_encoder_active | False |
| reranker_model | lexical-fallback |

## How to read this

- **Security is the only hard gate.** Isolation and adversarial checks must be 100%; a single failure invalidates the deployment regardless of retrieval quality.
- **Abstention categories pass by returning nothing.** A no-answer or isolation query that surfaces content has failed, even though it retrieved successfully.
- **Graph categories are the architectural justification.** If relationship and multi-hop queries score no better than semantic ones, the graph is cost without benefit.

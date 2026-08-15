# Retrieval Ablation Study

**Generated:** 2026-08-15T20:16:21.071398+00:00
**Queries:** 8 (8 with IR ground truth)

## Configuration comparison

| Configuration | R@1 | R@5 | R@10 | P@5 | MRR | NDCG@10 | P50 | P95 | P99 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vector | 0.900 | 1.000 | 1.000 | 0.300 | 1.000 | 1.000 | 467ms | 4246ms | 4246ms |
| Full+Reranker | 0.900 | 1.000 | 1.000 | 0.300 | 1.000 | 1.000 | 2459ms | 7310ms | 7310ms |

## Graph lift

Hybrid Recall@10 minus vector-only Recall@10. The number that decides whether the graph earns its cost, reported per category because the graph should help relationship and multi-hop questions specifically.

| Category | N | Vector R@10 | Hybrid R@10 | Lift | Solved only by graph |
| :--- | ---: | ---: | ---: | ---: | ---: |
| overall | 8 | 1.000 | 1.000 | **+0.000** | — |

## Ten slowest queries

| # | Query | Category | Cand. | Vector | BM25 | Graph | Rerank | Total |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `What are the symptoms of Cough?` | basic_semantic | 30 | 408ms | 11ms | 2364ms | 4508ms | **7310ms** |
| 2 | `How is Constipation managed?` | basic_semantic | 30 | 4105ms | 0ms | 0ms | 0ms | **4246ms** |
| 3 | `Tell me about Arthritis` | basic_semantic | 30 | 270ms | 0ms | 1385ms | 1703ms | **3380ms** |
| 4 | `What causes Fever?` | basic_semantic | 30 | 78ms | 0ms | 1183ms | 1537ms | **2828ms** |
| 5 | `What is Arrhythmia?` | basic_semantic | 30 | 123ms | 0ms | 501ms | 1853ms | **2505ms** |
| 6 | `Describe Adrenal Insufficiency` | basic_semantic | 30 | 121ms | 0ms | 25ms | 2281ms | **2459ms** |
| 7 | `How is Constipation managed?` | basic_semantic | 30 | 134ms | 0ms | 396ms | 1619ms | **2176ms** |
| 8 | `Information about Alzheimer's Disease` | basic_semantic | 30 | 99ms | 0ms | 430ms | 1590ms | **2140ms** |
| 9 | `What is Ashwagandha used for?` | basic_semantic | 6 | 61ms | 8ms | 79ms | 534ms | **717ms** |
| 10 | `Describe Adrenal Insufficiency` | basic_semantic | 30 | 514ms | 0ms | 0ms | 0ms | **660ms** |

## Environment

| Setting | Value |
| :--- | :--- |
| semantic_embeddings | True |
| embedding_model | bge-small-en-v1.5 |
| embedding_version | bge-small-en-v1.5/384/v1 |
| cross_encoder_active | True |
| reranker_model | ms-marco-MiniLM-L-6-v2 |

## Reading the ablation

- **F vs G** isolates RRF: identical paths, fusion the only difference.
- **G vs H** isolates the cross-encoder: identical retrieval and fusion, reranking the only difference.
- **A vs E** isolates the graph on otherwise identical vector retrieval.
- Quality and latency are reported together on purpose: optimising either alone produces a system that is fast and wrong, or accurate and unusable.

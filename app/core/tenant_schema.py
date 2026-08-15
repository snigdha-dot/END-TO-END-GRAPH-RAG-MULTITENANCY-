"""Per-tenant graph schema registry (plan section 2.2: AST & Schema Whitelisting).

Each tenant declares the vertex labels and relationship types its domain permits.
Traversal queries draw their relationship vocabulary from here rather than from a
single hardcoded list, and ingestion rejects anything outside the tenant's schema.

A movies knowledge base and an AI-trends knowledge base share no vocabulary; using
one list for both is what made traversal useless for real domains.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set

# ArcadeDB identifiers we are willing to emit into a query fragment. Anything not
# matching this is never interpolated, regardless of where it came from.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")


def is_safe_identifier(value: str) -> bool:
    """True when `value` is a conservative, interpolation-safe graph identifier."""
    return bool(_IDENTIFIER_RE.match(value or ""))


@dataclass(frozen=True)
class TenantGraphSchema:
    """The approved vocabulary for one tenant's knowledge base."""

    tenant_id: str
    display_name: str
    domain: str
    vertex_labels: Set[str]
    edge_types: Set[str]
    # NER labels handed to GLiNER at inference time (zero-shot, no retraining).
    ner_labels: List[str] = field(default_factory=list)
    # Edge types worth traversing by default, ordered most-informative first.
    default_traversal_edges: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for label in self.vertex_labels:
            if not is_safe_identifier(label):
                raise ValueError(f"Unsafe vertex label in schema '{self.tenant_id}': {label!r}")
        for etype in self.edge_types:
            if not is_safe_identifier(etype):
                raise ValueError(f"Unsafe edge type in schema '{self.tenant_id}': {etype!r}")

    @property
    def db_name(self) -> str:
        return f"tenant_{self.tenant_id.lower()}_kb"

    def validate_vertex_label(self, label: str) -> bool:
        return label in self.vertex_labels

    def validate_edge_type(self, edge_type: str) -> bool:
        return edge_type in self.edge_types

    def traversal_edges(self) -> List[str]:
        """Relationship vocabulary for multi-hop traversal in this domain."""
        if self.default_traversal_edges:
            return [e for e in self.default_traversal_edges if e in self.edge_types]
        return sorted(self.edge_types)

    def filter_edge_types(self, requested: List[str] | None) -> List[str]:
        """Intersect a caller's requested edge types with what this tenant permits."""
        if not requested:
            return self.traversal_edges()
        allowed = [e for e in requested if e in self.edge_types and is_safe_identifier(e)]
        return allowed or self.traversal_edges()


# --------------------------------------------------------------------- Registry

MOVIES_SCHEMA = TenantGraphSchema(
    tenant_id="movies_bot",
    display_name="Movies Knowledge Base",
    domain="film",
    vertex_labels={
        "Film", "Person", "Studio", "Genre", "Award", "Character", "Country", "Chunk",
    },
    edge_types={
        "DIRECTED", "ACTED_IN", "WROTE", "PRODUCED", "COMPOSED_FOR",
        "HAS_GENRE", "PRODUCED_BY", "WON_AWARD", "NOMINATED_FOR",
        "SEQUEL_OF", "PLAYED_CHARACTER", "RELEASED_IN", "MENTIONED_IN",
    },
    ner_labels=["Film", "Person", "Studio", "Genre", "Award", "Character", "Country"],
    default_traversal_edges=[
        "DIRECTED", "ACTED_IN", "WROTE", "PRODUCED", "HAS_GENRE",
        "PRODUCED_BY", "SEQUEL_OF", "WON_AWARD",
    ],
)

AI_TRENDS_SCHEMA = TenantGraphSchema(
    tenant_id="ai_trends_bot",
    display_name="AI Trends Knowledge Base",
    domain="ai_research",
    vertex_labels={
        "Model", "Organization", "Technique", "Paper", "Person",
        "Dataset", "Benchmark", "Hardware", "Chunk",
    },
    edge_types={
        "RELEASED_BY", "BUILDS_ON", "AUTHORED", "USES_TECHNIQUE",
        "TRAINED_ON", "EVALUATED_ON", "OUTPERFORMS", "CITES",
        "SUPERSEDES", "RUNS_ON", "AFFILIATED_WITH", "MENTIONED_IN",
    },
    ner_labels=[
        "Model", "Organization", "Technique", "Paper", "Person",
        "Dataset", "Benchmark", "Hardware",
    ],
    default_traversal_edges=[
        "BUILDS_ON", "RELEASED_BY", "USES_TECHNIQUE", "AUTHORED",
        "SUPERSEDES", "OUTPERFORMS", "TRAINED_ON", "CITES",
    ],
)

# --------------------------------------------------------------- generic schema
#
# A domain-neutral vocabulary that works for any dataset in any format, so a new
# tenant can be ingested without authoring a schema first.
#
# The labels are deliberately few and broad. Every additional label is a decision
# an extractor has to get right, and a wrong label produces a wrong canonical id
# (ids are label-scoped), so a small set of confidently-assigned labels beats a
# large set of guesses.
#
# The edge types cover the relation families that recur across domains:
#   AFFECTS         the treats/causes/influences family - the most common
#                   real-world relation and the one most queries follow
#   HAS_ATTRIBUTE   entity to a categorical value; how structured columns land
#   PART_OF         containment and membership
#   DERIVED_FROM    origin, source, derivation
#   ASSOCIATED_WITH co-occurrence where the verb is unknown
#   RELATED_TO      the untyped fallback, so a real relation is never discarded
#
# Trade-off worth stating: traversal is less precise than a domain schema. A query
# about treatments follows AFFECTS edges that also carry causes and side-effects,
# so recall rises and ranking suffers. Expect positive but modest graph lift.
# Promote frequent RELATED_TO/AFFECTS patterns into typed edges once the questions
# users actually ask are known.
GENERIC_VERTEX_LABELS = {
    "Entity",        # anything named that does not fit a narrower label
    "Person",        # people
    "Organization",  # companies, institutions, groups
    "Concept",       # abstract: techniques, conditions, categories, topics
    "Artifact",      # concrete named things: products, works, documents
    "Attribute",     # categorical values worth linking (severity, season, type)
    "Chunk",         # text units - required, the vector-search target
}

GENERIC_EDGE_TYPES = {
    "AFFECTS",          # treats, causes, influences, impacts
    "HAS_ATTRIBUTE",    # entity -> categorical value
    "PART_OF",          # containment, membership
    "DERIVED_FROM",     # origin, source
    "ASSOCIATED_WITH",  # co-occurrence, verb unknown
    "RELATED_TO",       # untyped fallback
    "CITES",            # references
    "PRECEDES",         # temporal or causal ordering
    "MENTIONED_IN",     # entity -> chunk; required, the graph<->text bridge
}

# Kept as aliases so existing callers continue to work.
DEFAULT_SCHEMA_LABELS = GENERIC_VERTEX_LABELS
DEFAULT_SCHEMA_EDGES = GENERIC_EDGE_TYPES


def build_generic_schema(tenant_id: str, display_name: str | None = None) -> TenantGraphSchema:
    """A domain-neutral schema usable by any tenant, for any data format."""
    return TenantGraphSchema(
        tenant_id=tenant_id,
        display_name=display_name or f"{tenant_id} Knowledge Base",
        domain="generic",
        vertex_labels=set(GENERIC_VERTEX_LABELS),
        edge_types=set(GENERIC_EDGE_TYPES),
        # Labels handed to GLiNER at inference time. Attribute and Chunk are
        # excluded: they come from structure, not from named-entity recognition.
        ner_labels=["Person", "Organization", "Concept", "Artifact"],
        # Ordered most-informative first, so traversal follows meaningful edges
        # before falling back to untyped association.
        default_traversal_edges=[
            "AFFECTS", "HAS_ATTRIBUTE", "PART_OF", "DERIVED_FROM",
            "ASSOCIATED_WITH", "CITES", "PRECEDES", "RELATED_TO",
        ],
    )


class TenantSchemaRegistry:
    """Lookup of tenant_id -> approved graph schema."""

    def __init__(self) -> None:
        self._schemas: Dict[str, TenantGraphSchema] = {}
        for schema in (MOVIES_SCHEMA, AI_TRENDS_SCHEMA):
            self._schemas[schema.tenant_id] = schema

    def register(self, schema: TenantGraphSchema) -> None:
        self._schemas[schema.tenant_id] = schema

    def has(self, tenant_id: str) -> bool:
        return tenant_id in self._schemas

    def get(self, tenant_id: str) -> TenantGraphSchema:
        """Return the tenant's schema, falling back to the generic one.

        An unregistered tenant is not an error: the generic schema lets any dataset
        be ingested without authoring a domain vocabulary first.
        """
        existing = self._schemas.get(tenant_id)
        if existing is not None:
            return existing
        return build_generic_schema(tenant_id)

    def all_tenants(self) -> List[str]:
        return sorted(self._schemas)


schema_registry = TenantSchemaRegistry()

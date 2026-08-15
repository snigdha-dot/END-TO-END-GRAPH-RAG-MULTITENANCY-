"""Hybrid entity & relationship extraction (plan section 3, Step 2).

Backends, in preference order:
  * GLiNER  - zero-shot NER. Entity labels come from the tenant's schema at
              inference time, so a movies KB extracts Film/Person/Studio while an
              AI-trends KB extracts Model/Organization/Technique, with no retraining.
  * spaCy   - faster, fixed label set, mapped onto the tenant schema.
  * regex   - dependency-free fallback.

Relations are extracted with schema-aware verb patterns built per tenant domain,
plus a proximity heuristic for co-occurring entities.

`LLMExtractionProvider` defines the interface the plan's LLM JSON-schema extractor
will implement. `NullLLMProvider` is active in this FOSS-only build; wiring a real
provider is a config change, not a refactor.
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.core.tenant_schema import TenantGraphSchema
from app.models.graph import Edge, Vertex
from app.services.relation_patterns import RelationExtractor
from app.services.type_inference import TypeInferencer

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- LLM stub
@dataclass
class LLMExtractionResult:
    vertices: List[Vertex]
    edges: List[Edge]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_name: str = "none"


class LLMExtractionProvider(ABC):
    """Interface for LLM JSON-schema extraction (plan section 3, Step 2)."""

    @abstractmethod
    async def extract(
        self, text: str, chunk_id: str, schema: TenantGraphSchema
    ) -> LLMExtractionResult: ...

    @abstractmethod
    async def disambiguate(
        self, mention: str, candidates: List[Dict[str, Any]], context: str
    ) -> Optional[str]: ...


class NullLLMProvider(LLMExtractionProvider):
    """No-op provider used when LLM_PROVIDER='none'. Keeps call sites uniform."""

    async def extract(
        self, text: str, chunk_id: str, schema: TenantGraphSchema
    ) -> LLMExtractionResult:
        return LLMExtractionResult(vertices=[], edges=[], model_name="none")

    async def disambiguate(
        self, mention: str, candidates: List[Dict[str, Any]], context: str
    ) -> Optional[str]:
        return None


def get_llm_provider() -> LLMExtractionProvider:
    """Resolve the configured LLM provider. Returns the null provider in FOSS mode."""
    if settings.LLM_PROVIDER == "none" or not settings.LLM_API_KEY:
        return NullLLMProvider()
    logger.warning(
        "LLM_PROVIDER=%s is configured but no concrete provider is implemented in this "
        "build; using NullLLMProvider.", settings.LLM_PROVIDER,
    )
    return NullLLMProvider()


# ------------------------------------------------------- domain relation patterns
# (verb phrase -> edge type). Applied only when the tenant's schema permits the edge.
_DOMAIN_RELATION_VERBS: Dict[str, List[Tuple[str, str]]] = {
    "film": [
        (r"directed(?:\s+by)?", "DIRECTED"),
        (r"starred(?:\s+in)?|acted\s+in|appears?\s+in|stars", "ACTED_IN"),
        (r"wrote|written\s+by|screenplay\s+by", "WROTE"),
        (r"produced(?:\s+by)?", "PRODUCED"),
        (r"composed(?:\s+the\s+score\s+for)?|scored", "COMPOSED_FOR"),
        (r"distributed\s+by|released\s+by", "PRODUCED_BY"),
        (r"won|received", "WON_AWARD"),
        (r"nominated\s+for", "NOMINATED_FOR"),
        (r"sequel\s+to|follows", "SEQUEL_OF"),
        (r"played|portrayed", "PLAYED_CHARACTER"),
    ],
    "ai_research": [
        (r"released\s+by|developed\s+by|created\s+by|introduced\s+by", "RELEASED_BY"),
        (r"builds?\s+on|based\s+on|extends?|derived\s+from", "BUILDS_ON"),
        (r"authored\s+by|written\s+by", "AUTHORED"),
        (r"uses?|employs?|leverages?|applies", "USES_TECHNIQUE"),
        (r"trained\s+on|pretrained\s+on|fine-?tuned\s+on", "TRAINED_ON"),
        (r"evaluated\s+on|benchmarked\s+on|tested\s+on", "EVALUATED_ON"),
        (r"outperforms?|beats|surpasses|exceeds", "OUTPERFORMS"),
        (r"cites?|references?", "CITES"),
        (r"supersedes?|replaces?|succeeded\s+by", "SUPERSEDES"),
        (r"runs?\s+on|deployed\s+on", "RUNS_ON"),
    ],
    "generic": [
        (r"depends?\s+on|requires?|relies\s+on", "DEPENDS_ON"),
        (r"owns?", "OWNS"),
        (r"manages?|maintains?", "MANAGES"),
        (r"cites?|references?", "CITES"),
        (r"contains?|includes?|has\s+part", "HAS_PART"),
        (r"related\s+to|associated\s+with", "RELATED_TO"),
    ],
}

# spaCy's fixed labels mapped onto domain schemas.
_SPACY_LABEL_MAP: Dict[str, Dict[str, str]] = {
    "film": {"PERSON": "Person", "ORG": "Studio", "WORK_OF_ART": "Film", "GPE": "Country"},
    "ai_research": {"PERSON": "Person", "ORG": "Organization", "PRODUCT": "Model", "WORK_OF_ART": "Paper"},
    "generic": {"PERSON": "Person", "ORG": "Organization", "PRODUCT": "Concept", "GPE": "Concept"},
}

_STOPWORD_TITLES = {
    "the", "a", "an", "this", "that", "these", "those", "it", "its",
    "introduction", "overview", "summary", "conclusion", "references", "contents",
    "abstract", "background", "plot", "cast", "reception", "production", "notes",
}

# Sentence openers, connectives, and section furniture that survive the
# capitalization heuristic but are never entities.
_NON_ENTITY_WORDS = {
    "however", "although", "therefore", "moreover", "furthermore", "meanwhile",
    "nevertheless", "additionally", "consequently", "subsequently", "similarly",
    "instead", "despite", "while", "when", "where", "after", "before", "during",
    "following", "according", "based", "using", "several", "many", "most", "some",
    "other", "another", "both", "each", "such", "these", "those", "there", "here",
    "他", "his", "her", "their", "they", "he", "she", "we", "you", "who", "what",
    "in", "on", "at", "to", "for", "with", "by", "from", "as", "is", "was", "were",
    "development", "release", "critical", "commercial", "critical response",
    "see also", "external links", "further reading", "citation", "retrieved",
    "archived", "original", "isbn", "doi", "pp", "vol", "ed", "et al",
}

# A predicate that marks the preceding capitalized word as a real subject rather
# than a sentence opener: "Inception is a 2010 film", "GPT-4 was released by...".
_SUBJECT_PREDICATE = re.compile(
    r"^\s+(?:is|was|are|were|has|have|had|became|remains|features|stars|"
    r"received|premiered|grossed|introduced|outperforms|builds|uses|supersedes|"
    r"consists|refers|marks|won|earned)\b",
    re.IGNORECASE,
)

# Dates, standalone years, and month names.
_DATE_LIKE = re.compile(
    r"^(?:\d{1,4}|January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Monday|Tuesday|Wednesday|Thursday|Friday|"
    r"Saturday|Sunday)(?:\s+\d{1,4})?$",
    re.IGNORECASE,
)


def normalize_entity_name(name: str) -> str:
    """Canonical form used for matching: lowercase, punctuation-stripped, underscored."""
    cleaned = re.sub(r"[^\w\s-]", "", (name or "").strip().lower())
    cleaned = re.sub(r"[\s-]+", "_", cleaned)
    return cleaned.strip("_")


def entity_id_for(name: str, label: str = "") -> str:
    """Pre-resolution mention id.

    Label-scoped so two entities of different types sharing a surface form stay
    distinct all the way through resolution.
    """
    normalized = normalize_entity_name(name)
    return f"canon_{label.lower()}_{normalized}" if label else f"canon_{normalized}"


class ExtractionService:
    """Extracts schema-conformant vertices and edges from chunk text."""

    def __init__(self) -> None:
        self._gliner = None
        self._spacy = None
        self._backend_attempted = False
        self._active_backend = "regex"
        self._llm = get_llm_provider()

    # ------------------------------------------------------------------ backends
    def _load_backend(self) -> None:
        if self._backend_attempted:
            return
        self._backend_attempted = True
        desired = settings.NER_BACKEND

        if desired == "gliner":
            try:
                from gliner import GLiNER  # noqa: PLC0415

                logger.info("Loading GLiNER model '%s'...", settings.GLINER_MODEL)
                self._gliner = GLiNER.from_pretrained(settings.GLINER_MODEL)
                self._active_backend = "gliner"
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("GLiNER unavailable (%s); trying spaCy.", exc)
                desired = "spacy"

        if desired == "spacy":
            try:
                import spacy  # noqa: PLC0415

                self._spacy = spacy.load(settings.SPACY_MODEL)
                self._active_backend = "spacy"
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("spaCy unavailable (%s); using regex extraction.", exc)

        self._active_backend = "regex"

    @property
    def active_backend(self) -> str:
        self._load_backend()
        return self._active_backend

    @property
    def model_label(self) -> str:
        backend = self.active_backend
        if backend == "gliner":
            return settings.GLINER_MODEL.split("/")[-1]
        if backend == "spacy":
            return f"spacy-{settings.SPACY_MODEL}"
        return "regex-extractor"

    # ------------------------------------------------------------------ entities
    def _extract_entities(self, text: str, schema: TenantGraphSchema) -> List[Dict[str, Any]]:
        self._load_backend()
        if self._active_backend == "gliner":
            return self._extract_gliner(text, schema)
        if self._active_backend == "spacy":
            return self._extract_spacy(text, schema)
        return self._extract_regex(text, schema)

    def _extract_gliner(self, text: str, schema: TenantGraphSchema) -> List[Dict[str, Any]]:
        """Zero-shot extraction using the tenant's own label set."""
        labels = [l for l in schema.ner_labels if l in schema.vertex_labels]
        if not labels:
            return self._extract_regex(text, schema)
        try:
            predictions = self._gliner.predict_entities(text, labels, threshold=0.45)
        except Exception as exc:  # noqa: BLE001
            logger.warning("GLiNER inference failed (%s); falling back to regex.", exc)
            return self._extract_regex(text, schema)

        out: List[Dict[str, Any]] = []
        for pred in predictions:
            name = str(pred.get("text", "")).strip()
            label = str(pred.get("label", "")).strip()
            if len(name) < 2 or not schema.validate_vertex_label(label):
                continue
            out.append(
                {
                    "name": name,
                    "label": label,
                    "confidence": float(pred.get("score", 0.5)),
                    "start": int(pred.get("start", -1)),
                    "end": int(pred.get("end", -1)),
                }
            )
        return out

    def _extract_spacy(self, text: str, schema: TenantGraphSchema) -> List[Dict[str, Any]]:
        mapping = _SPACY_LABEL_MAP.get(schema.domain, _SPACY_LABEL_MAP["generic"])
        try:
            doc = self._spacy(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("spaCy inference failed (%s); falling back to regex.", exc)
            return self._extract_regex(text, schema)

        out: List[Dict[str, Any]] = []
        for ent in doc.ents:
            label = mapping.get(ent.label_)
            if not label or not schema.validate_vertex_label(label):
                continue
            name = ent.text.strip()
            if len(name) < 2:
                continue
            out.append(
                {"name": name, "label": label, "confidence": 0.75,
                 "start": ent.start_char, "end": ent.end_char}
            )
        return out

    def _extract_regex(self, text: str, schema: TenantGraphSchema) -> List[Dict[str, Any]]:
        """Capitalized-phrase heuristic with contextual type inference.

        Every occurrence is kept (not just the first), because relation extraction
        needs each mention's position to pair entities with the verb between them.
        Types come from `TypeInferencer` rather than a single default: a shared
        default made every entity the same label, which broke label-scoped ids and
        caused schema-valid relations to be rejected.
        """
        default_label = self._default_label(schema)
        raw: List[Dict[str, Any]] = []

        # Greedily match consecutive capitalized tokens as ONE mention, optionally
        # joined by a lowercase connective ("Warner Bros. of America"). Matching
        # token-by-token split "Christopher Nolan" into two separate Person nodes.
        for match in re.finditer(
            # `[ \t]` rather than `\s`: a newline ends a mention, and a trailing
            # period ends it too, so "Michael Caine.\nAfter" is one entity, not two
            # words glued together.
            r"\b[A-Z][a-zA-Z0-9'’\-]*"
            r"(?:[ \t]+(?:of|the|and|de|von|van|di|da)[ \t]+[A-Z][a-zA-Z0-9'’\-]*"
            r"|[ \t]+[A-Z][a-zA-Z0-9'’\-]*){0,4}",
            text,
        ):
            name = match.group(0).strip().rstrip(",;:.")
            if not self._plausible_entity(name, text, match.start()):
                continue
            raw.append(
                {
                    "name": name,
                    "label": default_label,
                    "confidence": 0.55,
                    "start": match.start(),
                    "end": match.start() + len(name),
                }
            )

        if not raw:
            return []

        inferencer = TypeInferencer(schema)
        return inferencer.infer_batch(raw, text, default_label)

    @staticmethod
    def _plausible_entity(name: str, text: str, start: int) -> bool:
        """Reject the noise that dominates real prose.

        Encyclopedic text is full of capitalized tokens that are not entities:
        sentence-initial words, headings, month names, and citation furniture.
        Without this filter the graph fills with 'The', 'However', and 'January'.
        """
        if len(name) < 3 or len(name) > 60:
            return False

        lowered = name.lower()
        if lowered in _STOPWORD_TITLES or lowered in _NON_ENTITY_WORDS:
            return False
        if _DATE_LIKE.match(name):
            return False
        # All-caps runs of 1-2 characters are usually initials or artefacts.
        if name.isupper() and len(name) <= 2:
            return False

        # A single capitalized word at a sentence boundary is usually a sentence
        # opener -- but not always: "Inception is a 2010 film" opens with the very
        # entity the document is about. Rejecting all of them lost every subject
        # entity, and with it every relation that needed one as an endpoint.
        # Keep it when the following text defines or predicates it.
        if " " not in name:
            preceding = text[max(0, start - 2) : start]
            at_boundary = start == 0 or preceding.strip().endswith((".", "!", "?", "\n"))
            if at_boundary and not _SUBJECT_PREDICATE.match(text[start + len(name) :]):
                return False
        return True

    @staticmethod
    def _default_label(schema: TenantGraphSchema) -> str:
        for candidate in ("Entity", "Concept", "Person"):
            if candidate in schema.vertex_labels:
                return candidate
        return sorted(schema.vertex_labels - {"Chunk"})[0] if schema.vertex_labels else "Entity"

    # ------------------------------------------------------------------ relations
    def _extract_relations(
        self, text: str, entities: List[Dict[str, Any]], schema: TenantGraphSchema
    ) -> List[Dict[str, Any]]:
        """Find typed relations between entity mentions.

        Delegates to `RelationExtractor`, which handles the constructions real prose
        actually uses: passive voice with an agentive `by`, coordinated object lists,
        and sentence-bounded pairing. The previous nearest-neighbour heuristic
        produced 3 edges from a 417,000-character corpus.
        """
        positioned = [e for e in entities if e.get("start", -1) >= 0]
        if len(positioned) < 2:
            return []

        extractor = RelationExtractor(schema.domain, schema.edge_types)
        return [
            {
                "source": relation.source_name,
                "target": relation.target_name,
                "type": relation.edge_type,
                "confidence": relation.confidence,
                "evidence": relation.evidence,
            }
            for relation in extractor.extract(text, positioned)
        ]

    # ------------------------------------------------------------------ public
    def extract_from_chunk(
        self, text: str, chunk_id: str, schema: TenantGraphSchema
    ) -> Tuple[List[Vertex], List[Edge]]:
        """Extract schema-conformant vertices and edges from one chunk."""
        if not text or not text.strip():
            return [], []

        raw_entities = self._extract_entities(text, schema)

        # Keep the strongest mentions per chunk. Real prose produces a long tail of
        # low-confidence capitalized fragments; each one costs an entity write and a
        # MENTIONED_IN write while adding no retrievable signal. Relations are
        # extracted from the full mention list first, so nothing that participates
        # in an edge is lost to this cap.
        relations = self._extract_relations(text, raw_entities, schema)
        related_names = {
            normalize_entity_name(r["source"]) for r in relations
        } | {normalize_entity_name(r["target"]) for r in relations}

        ranked = sorted(
            raw_entities,
            key=lambda e: (
                normalize_entity_name(e["name"]) in related_names,
                e.get("confidence", 0.0),
            ),
            reverse=True,
        )
        keep = {normalize_entity_name(e["name"]) for e in ranked[: settings.MAX_ENTITIES_PER_CHUNK]}
        keep |= related_names

        vertices: List[Vertex] = []
        by_key: Dict[Tuple[str, str], Vertex] = {}
        for ent in raw_entities:
            normalized = normalize_entity_name(ent["name"])
            key = (normalized, ent["label"])
            if not normalized or key in by_key or normalized not in keep:
                continue
            vertex = Vertex(
                id=entity_id_for(ent["name"], ent["label"]),
                label=ent["label"],
                properties={
                    "name": ent["name"],
                    "normalized_name": normalized,
                    "provenance": chunk_id,
                    "confidence": ent.get("confidence", 0.5),
                    "extractor": self.model_label,
                },
            )
            by_key[key] = vertex
            vertices.append(vertex)

        # Canonical ids are label-scoped, so an edge endpoint must resolve to the
        # same label the vertex was created with. Looking it up here keeps edges
        # pointing at nodes that actually exist.
        label_by_name = {
            normalize_entity_name(e["name"]): e["label"] for e in raw_entities
        }

        edges: List[Edge] = []
        seen_edges: set[Tuple[str, str, str]] = set()
        for rel in relations:
            if rel["confidence"] < settings.EDGE_CONFIDENCE_THRESHOLD:
                continue
            src_norm = normalize_entity_name(rel["source"])
            tgt_norm = normalize_entity_name(rel["target"])
            src_label = label_by_name.get(src_norm)
            tgt_label = label_by_name.get(tgt_norm)
            if not src_label or not tgt_label:
                continue
            src_id = entity_id_for(rel["source"], src_label)
            tgt_id = entity_id_for(rel["target"], tgt_label)
            key = (src_id, rel["type"], tgt_id)
            if src_id == tgt_id or key in seen_edges:
                continue
            seen_edges.add(key)
            edges.append(
                Edge(
                    source=src_id,
                    target=tgt_id,
                    type=rel["type"],
                    properties={
                        "confidence": rel["confidence"],
                        "chunk_id": chunk_id,
                        "evidence": rel["evidence"],
                    },
                )
            )

        return vertices, edges


extraction_service = ExtractionService()

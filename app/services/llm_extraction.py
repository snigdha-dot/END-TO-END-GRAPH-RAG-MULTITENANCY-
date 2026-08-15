"""LLM entity and relation extraction under schema constraint.

The pattern is constrained proposal, not free-form induction. Microsoft GraphRAG
lets the LLM invent types per document, which produces rich graphs that are
inconsistent: the same concept is typed `Herb` in one chunk and `Medicine` in the
next, and nothing links them.

Here the LLM proposes, and the schema decides:

    LLM reads chunk -> proposes typed entities and relations
                    -> validator accepts, coerces to the nearest valid type,
                       or rejects
                    -> only schema-conformant output reaches the graph

That matters beyond tidiness. Canonical ids are label-scoped
(`canon_{label}_{name}`), so a drifting type produces a different id for the same
real-world entity, and the graph fragments into duplicates that never connect.

Providers are pluggable. `NullLLMProvider` is active in the FOSS-only build, so
every call site works unchanged whether or not a provider is configured.
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.core.tenant_schema import TenantGraphSchema
from app.models.graph import Edge, Vertex

logger = logging.getLogger(__name__)


@dataclass
class ExtractionProposal:
    """What an LLM proposed, before schema validation."""

    entities: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_name: str = "none"


@dataclass
class ValidationOutcome:
    """What survived validation, and what did not."""

    entities: List[Vertex] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    coerced: int = 0
    rejected: int = 0
    rejections: List[str] = field(default_factory=list)


EXTRACTION_PROMPT = """You extract a knowledge graph from text.

Allowed entity types (use ONLY these):
{entity_types}

Allowed relation types (use ONLY these):
{relation_types}

Rules:
- Extract only entities that are explicitly named in the text.
- Use the exact surface form as it appears; do not paraphrase names.
- Every relation's source and target must be entities you also extracted.
- If a relation does not fit an allowed type, omit it rather than forcing it.
- Assign confidence between 0 and 1 reflecting how explicitly the text states it.

Return ONLY valid JSON, no commentary:
{{"entities": [{{"name": "...", "type": "...", "confidence": 0.9}}],
  "relations": [{{"source": "...", "target": "...", "type": "...",
                  "confidence": 0.9, "evidence": "..."}}]}}

Text:
{text}
"""


class LLMExtractionProvider(ABC):
    """Interface every extraction backend implements."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    async def propose(self, text: str, schema: TenantGraphSchema) -> ExtractionProposal: ...


class NullLLMProvider(LLMExtractionProvider):
    """Active in the FOSS-only build. Proposes nothing; call sites stay uniform."""

    @property
    def model_name(self) -> str:
        return "none"

    @property
    def is_available(self) -> bool:
        return False

    async def propose(self, text: str, schema: TenantGraphSchema) -> ExtractionProposal:
        return ExtractionProposal()


class HTTPLLMProvider(LLMExtractionProvider):
    """Generic OpenAI-compatible or Gemini extraction provider.

    Implemented against the HTTP APIs directly rather than a vendor SDK, so
    enabling extraction is a config change and adds no dependency.
    """

    def __init__(self, provider: str, api_key: str, model: str) -> None:
        self.provider = provider
        self.api_key = api_key
        self.model = model

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    async def propose(self, text: str, schema: TenantGraphSchema) -> ExtractionProposal:
        import httpx  # noqa: PLC0415

        prompt = EXTRACTION_PROMPT.format(
            entity_types=", ".join(sorted(schema.vertex_labels - {"Chunk"})),
            relation_types=", ".join(sorted(schema.edge_types - {"MENTIONED_IN"})),
            text=text[:6000],
        )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                if self.provider == "gemini":
                    payload, url, headers = self._gemini_request(prompt)
                else:
                    payload, url, headers = self._openai_request(prompt)

                response = await client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    logger.warning(
                        "LLM extraction failed (HTTP %s): %s",
                        response.status_code, response.text[:200],
                    )
                    return ExtractionProposal(model_name=self.model)

                content, prompt_tokens, completion_tokens = self._parse_response(
                    response.json()
                )
        except Exception as exc:  # noqa: BLE001 - extraction must not fail ingestion
            logger.warning("LLM extraction error: %s", exc)
            return ExtractionProposal(model_name=self.model)

        parsed = self._parse_json(content)
        return ExtractionProposal(
            entities=parsed.get("entities", []),
            relations=parsed.get("relations", []),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_name=self.model,
        )

    def _openai_request(self, prompt: str) -> Tuple[Dict, str, Dict]:
        return (
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            "https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {self.api_key}"},
        )

    def _gemini_request(self, prompt: str) -> Tuple[Dict, str, Dict]:
        return (
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
            },
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}",
            {},
        )

    def _parse_response(self, data: Dict) -> Tuple[str, int, int]:
        if self.provider == "gemini":
            candidates = data.get("candidates", [])
            content = ""
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                content = parts[0].get("text", "") if parts else ""
            usage = data.get("usageMetadata", {})
            return (
                content,
                usage.get("promptTokenCount", 0),
                usage.get("candidatesTokenCount", 0),
            )

        choices = data.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        usage = data.get("usage", {})
        return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    @staticmethod
    def _parse_json(content: str) -> Dict[str, Any]:
        """Parse the model's JSON, tolerating markdown fences around it."""
        if not content:
            return {}
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.MULTILINE)
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            # Recover the outermost object if the model wrapped it in prose.
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            logger.warning("LLM returned unparseable JSON: %s", content[:160])
            return {}


class SchemaValidator:
    """Turns proposals into schema-conformant vertices and edges.

    Coercion before rejection: an LLM that says `Medicine` when the schema has
    `Artifact` is right about the entity and wrong about the label, so mapping it
    is better than discarding a real extraction.
    """

    # Common near-misses, mapped onto the generic vocabulary.
    _TYPE_SYNONYMS: Dict[str, str] = {
        "people": "Person", "human": "Person", "individual": "Person",
        "author": "Person", "doctor": "Person", "patient": "Person",
        "company": "Organization", "institution": "Organization",
        "corporation": "Organization", "org": "Organization", "lab": "Organization",
        "product": "Artifact", "document": "Artifact", "work": "Artifact",
        "medicine": "Artifact", "drug": "Artifact", "formulation": "Artifact",
        "book": "Artifact", "paper": "Artifact", "model": "Artifact",
        "idea": "Concept", "topic": "Concept", "theme": "Concept",
        "condition": "Concept", "disease": "Concept", "symptom": "Concept",
        "technique": "Concept", "method": "Concept", "process": "Concept",
        "property": "Attribute", "category": "Attribute", "value": "Attribute",
        "type": "Attribute", "status": "Attribute", "level": "Attribute",
        "thing": "Entity", "object": "Entity", "item": "Entity",
    }

    _RELATION_SYNONYMS: Dict[str, str] = {
        "treats": "AFFECTS", "causes": "AFFECTS", "influences": "AFFECTS",
        "impacts": "AFFECTS", "prevents": "AFFECTS", "cures": "AFFECTS",
        "has": "HAS_ATTRIBUTE", "has_property": "HAS_ATTRIBUTE",
        "is_a": "HAS_ATTRIBUTE", "type_of": "HAS_ATTRIBUTE",
        "contains": "PART_OF", "includes": "PART_OF", "member_of": "PART_OF",
        "belongs_to": "PART_OF", "component_of": "PART_OF",
        "from": "DERIVED_FROM", "source": "DERIVED_FROM",
        "based_on": "DERIVED_FROM", "extracted_from": "DERIVED_FROM",
        "references": "CITES", "mentions": "CITES",
        "before": "PRECEDES", "followed_by": "PRECEDES", "leads_to": "PRECEDES",
        "related": "RELATED_TO", "linked_to": "ASSOCIATED_WITH",
        "associated": "ASSOCIATED_WITH", "correlated_with": "ASSOCIATED_WITH",
    }

    def validate(
        self,
        proposal: ExtractionProposal,
        schema: TenantGraphSchema,
        chunk_id: str,
        min_confidence: Optional[float] = None,
    ) -> ValidationOutcome:
        from app.services.extraction_service import (  # noqa: PLC0415
            entity_id_for,
            normalize_entity_name,
        )

        threshold = (
            min_confidence if min_confidence is not None else settings.EDGE_CONFIDENCE_THRESHOLD
        )
        outcome = ValidationOutcome()
        label_by_name: Dict[str, str] = {}

        for raw in proposal.entities:
            name = str(raw.get("name", "")).strip()
            proposed_type = str(raw.get("type", "")).strip()
            if not name or len(name) > 120:
                outcome.rejected += 1
                continue

            label, was_coerced = self._resolve_label(proposed_type, schema)
            if label is None:
                outcome.rejected += 1
                outcome.rejections.append(f"entity type {proposed_type!r} not in schema")
                continue
            if was_coerced:
                outcome.coerced += 1

            normalized = normalize_entity_name(name)
            if not normalized:
                outcome.rejected += 1
                continue

            label_by_name[normalized] = label
            outcome.entities.append(
                Vertex(
                    id=entity_id_for(name, label),
                    label=label,
                    properties={
                        "name": name,
                        "normalized_name": normalized,
                        "entity_id": entity_id_for(name, label),
                        "confidence": float(raw.get("confidence", 0.8)),
                        "provenance": chunk_id,
                        "extractor": proposal.model_name,
                    },
                )
            )

        for raw in proposal.relations:
            source = str(raw.get("source", "")).strip()
            target = str(raw.get("target", "")).strip()
            proposed_type = str(raw.get("type", "")).strip()
            confidence = float(raw.get("confidence", 0.8))

            if confidence < threshold:
                outcome.rejected += 1
                continue

            edge_type, was_coerced = self._resolve_edge(proposed_type, schema)
            if edge_type is None:
                outcome.rejected += 1
                outcome.rejections.append(f"relation type {proposed_type!r} not in schema")
                continue
            if was_coerced:
                outcome.coerced += 1

            source_norm = normalize_entity_name(source)
            target_norm = normalize_entity_name(target)
            source_label = label_by_name.get(source_norm)
            target_label = label_by_name.get(target_norm)

            # An endpoint the model did not also extract would dangle.
            if not source_label or not target_label or source_norm == target_norm:
                outcome.rejected += 1
                continue

            outcome.edges.append(
                Edge(
                    source=entity_id_for(source, source_label),
                    target=entity_id_for(target, target_label),
                    type=edge_type,
                    properties={
                        "confidence": confidence,
                        "chunk_id": chunk_id,
                        "evidence": str(raw.get("evidence", ""))[:500],
                        "extractor": proposal.model_name,
                    },
                )
            )

        return outcome

    def _resolve_label(
        self, proposed: str, schema: TenantGraphSchema
    ) -> Tuple[Optional[str], bool]:
        if not proposed:
            return ("Entity", True) if "Entity" in schema.vertex_labels else (None, False)

        if proposed in schema.vertex_labels:
            return proposed, False

        for label in schema.vertex_labels:
            if label.lower() == proposed.lower():
                return label, True

        mapped = self._TYPE_SYNONYMS.get(proposed.lower().replace(" ", "_"))
        if mapped and mapped in schema.vertex_labels:
            return mapped, True

        # Last resort: keep the entity under the catch-all rather than lose it.
        if "Entity" in schema.vertex_labels:
            return "Entity", True
        return None, False

    def _resolve_edge(
        self, proposed: str, schema: TenantGraphSchema
    ) -> Tuple[Optional[str], bool]:
        if not proposed:
            return (None, False)

        normalized = proposed.upper().replace(" ", "_").replace("-", "_")
        if normalized in schema.edge_types:
            return normalized, False

        mapped = self._RELATION_SYNONYMS.get(proposed.lower().replace(" ", "_"))
        if mapped and mapped in schema.edge_types:
            return mapped, True

        # Unlike entities, an unmappable relation is dropped rather than coerced:
        # a wrong relation type asserts something false about the world, whereas a
        # wrong entity label only makes it harder to find.
        return None, False


def get_llm_provider() -> LLMExtractionProvider:
    """Resolve the configured provider, or the null provider in FOSS-only mode."""
    if settings.LLM_PROVIDER == "none" or not settings.LLM_API_KEY:
        return NullLLMProvider()
    return HTTPLLMProvider(
        provider=settings.LLM_PROVIDER,
        api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL,
    )


schema_validator = SchemaValidator()

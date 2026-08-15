"""Relation extraction patterns for real prose.

The first version matched only `X verb Y` in active voice, which produced 3 edges
from a 417,000-character corpus. Encyclopedic writing is mostly not shaped that
way. It writes:

    "Inception is a 2010 film written and directed by Christopher Nolan"
      -> passive, with the agent AFTER the verb, and two verbs coordinated
    "Nolan, who also directed Interstellar and Dunkirk, ..."
      -> relative clause, with a coordinated object list
    "Hans Zimmer, the composer, worked on Dune"
      -> appositive

This module handles those four shapes. Each returns (subject, object) in the
correct direction: passive constructions invert, so `X directed by Y` must yield
`Y DIRECTED X`, not the reverse.

Label constraints come from the tenant schema, so `Person DIRECTED Film` is
accepted while `Person DIRECTED Person` is rejected as a schema violation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class RelationRule:
    """One relation pattern and the entity types it may connect."""

    edge_type: str
    verbs: str                      # regex alternation of verb forms
    passive: bool = False           # "X was directed by Y" -> Y -> X
    subject_labels: Tuple[str, ...] = ()
    object_labels: Tuple[str, ...] = ()
    confidence: float = 0.85

    def accepts(self, subject_label: str, object_label: str) -> bool:
        if self.subject_labels and subject_label not in self.subject_labels:
            return False
        if self.object_labels and object_label not in self.object_labels:
            return False
        return True


# --------------------------------------------------------------------- film
FILM_RULES: List[RelationRule] = [
    RelationRule("DIRECTED", r"directed", passive=True,
                 subject_labels=("Person",), object_labels=("Film",), confidence=0.92),
    RelationRule("DIRECTED", r"directed|helmed",
                 subject_labels=("Person",), object_labels=("Film",), confidence=0.90),
    RelationRule("WROTE", r"written|co-written|scripted", passive=True,
                 subject_labels=("Person",), object_labels=("Film",), confidence=0.90),
    RelationRule("WROTE", r"wrote|co-wrote",
                 subject_labels=("Person",), object_labels=("Film",), confidence=0.88),
    RelationRule("PRODUCED", r"produced|co-produced", passive=True,
                 subject_labels=("Person", "Studio"), object_labels=("Film",), confidence=0.88),
    RelationRule("PRODUCED", r"produced|co-produced",
                 subject_labels=("Person", "Studio"), object_labels=("Film",), confidence=0.86),
    RelationRule("ACTED_IN", r"starring|featuring",
                 subject_labels=("Film",), object_labels=("Person",), confidence=0.88),
    RelationRule("ACTED_IN", r"stars|starred|appears|appeared|portrays|portrayed|plays|played",
                 subject_labels=("Film", "Person"), object_labels=("Person", "Film", "Character"),
                 confidence=0.85),
    RelationRule("COMPOSED_FOR", r"composed|scored", passive=True,
                 subject_labels=("Person",), object_labels=("Film",), confidence=0.90),
    RelationRule("COMPOSED_FOR", r"composed|scored",
                 subject_labels=("Person",), object_labels=("Film",), confidence=0.88),
    RelationRule("PRODUCED_BY", r"distributed|released", passive=True,
                 subject_labels=("Studio",), object_labels=("Film",), confidence=0.86),
    RelationRule("HAS_GENRE", r"is a|genre",
                 subject_labels=("Film",), object_labels=("Genre",), confidence=0.80),
    RelationRule("WON_AWARD", r"won|received",
                 subject_labels=("Film", "Person"), object_labels=("Award",), confidence=0.85),
    RelationRule("NOMINATED_FOR", r"nominated",
                 subject_labels=("Film", "Person"), object_labels=("Award",), confidence=0.85),
    RelationRule("SEQUEL_OF", r"sequel|follows|follow-up",
                 subject_labels=("Film",), object_labels=("Film",), confidence=0.84),
]

# ------------------------------------------------------------- ai research
AI_RULES: List[RelationRule] = [
    RelationRule("RELEASED_BY", r"released|developed|created|introduced|announced|published",
                 passive=True, subject_labels=("Organization",),
                 object_labels=("Model", "Technique", "Paper", "Dataset"), confidence=0.90),
    RelationRule("RELEASED_BY", r"released|developed|created|introduced|announced|unveiled",
                 subject_labels=("Organization",),
                 object_labels=("Model", "Technique", "Paper", "Dataset"), confidence=0.88),
    RelationRule("BUILDS_ON", r"builds on|built on|based on|derived from|extends|adapted from",
                 subject_labels=("Model", "Technique"),
                 object_labels=("Model", "Technique"), confidence=0.88),
    RelationRule("USES_TECHNIQUE", r"uses|using|employs|leverages|applies|relies on|incorporates",
                 subject_labels=("Model", "Technique"),
                 object_labels=("Technique",), confidence=0.86),
    RelationRule("TRAINED_ON", r"trained|pretrained|pre-trained|fine-tuned", passive=True,
                 subject_labels=("Dataset",), object_labels=("Model",), confidence=0.88),
    RelationRule("TRAINED_ON", r"trained on|pretrained on|fine-tuned on",
                 subject_labels=("Model",), object_labels=("Dataset",), confidence=0.88),
    RelationRule("EVALUATED_ON", r"evaluated on|benchmarked on|tested on|measured on",
                 subject_labels=("Model",), object_labels=("Benchmark", "Dataset"), confidence=0.86),
    RelationRule("OUTPERFORMS", r"outperforms|surpasses|beats|exceeds|improves on",
                 subject_labels=("Model", "Technique"),
                 object_labels=("Model", "Technique"), confidence=0.87),
    RelationRule("SUPERSEDES", r"supersedes|replaces|succeeded by|successor to",
                 subject_labels=("Model",), object_labels=("Model",), confidence=0.86),
    RelationRule("AUTHORED", r"authored|written", passive=True,
                 subject_labels=("Person",), object_labels=("Paper",), confidence=0.88),
    RelationRule("CITES", r"cites|references|builds upon",
                 subject_labels=("Paper",), object_labels=("Paper",), confidence=0.82),
    RelationRule("RUNS_ON", r"runs on|deployed on|trained using",
                 subject_labels=("Model",), object_labels=("Hardware",), confidence=0.84),
    RelationRule("AFFILIATED_WITH", r"works at|joined|affiliated with|researcher at|founded",
                 subject_labels=("Person",), object_labels=("Organization",), confidence=0.85),
]

GENERIC_RULES: List[RelationRule] = [
    RelationRule("DEPENDS_ON", r"depends on|requires|relies on|needs", confidence=0.84),
    RelationRule("MANAGES", r"manages|maintains|oversees", confidence=0.84),
    RelationRule("OWNS", r"owns|possesses", confidence=0.84),
    RelationRule("CITES", r"cites|references", confidence=0.82),
    RelationRule("HAS_PART", r"contains|includes|comprises", confidence=0.82),
    RelationRule("RELATED_TO", r"related to|associated with|linked to", confidence=0.75),
]

RULES_BY_DOMAIN: Dict[str, List[RelationRule]] = {
    "film": FILM_RULES,
    "ai_research": AI_RULES,
    "generic": GENERIC_RULES,
}

# Sentence boundary that tolerates abbreviations ("Warner Bros. released").
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
# Coordination inside an object list: "Interstellar, Dunkirk and Oppenheimer".
_COORDINATION = re.compile(r",\s*|\s+and\s+|\s+&\s+", re.IGNORECASE)


@dataclass
class ExtractedRelation:
    source_name: str
    target_name: str
    edge_type: str
    confidence: float
    evidence: str


class RelationExtractor:
    """Extracts typed relations from sentences containing known entity mentions."""

    def __init__(self, domain: str, allowed_edges: Set[str]) -> None:
        self.rules = [
            rule for rule in RULES_BY_DOMAIN.get(domain, GENERIC_RULES)
            if rule.edge_type in allowed_edges
        ]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _sentences(text: str) -> List[Tuple[int, str]]:
        """Split into sentences, keeping each one's offset in the original text."""
        out: List[Tuple[int, str]] = []
        position = 0
        for part in _SENTENCE_SPLIT.split(text):
            index = text.find(part, position)
            if index < 0:
                index = position
            out.append((index, part))
            position = index + len(part)
        return out

    @staticmethod
    def _mentions_in(
        mentions: Sequence[Dict], start: int, end: int
    ) -> List[Dict]:
        return sorted(
            [m for m in mentions if m.get("start", -1) >= start and m.get("end", 0) <= end],
            key=lambda m: m["start"],
        )

    def _emit(
        self,
        rule: RelationRule,
        left: Dict,
        right: Dict,
        evidence: str,
        distance_penalty: float = 0.0,
    ) -> Optional[ExtractedRelation]:
        """Build a relation in the direction the rule specifies, if labels allow."""
        # Passive voice inverts: "X directed by Y" means Y directed X.
        subject, obj = (right, left) if rule.passive else (left, right)

        if not rule.accepts(subject["label"], obj["label"]):
            return None
        if subject["name"].lower() == obj["name"].lower():
            return None

        confidence = max(0.5, rule.confidence - distance_penalty)
        return ExtractedRelation(
            source_name=subject["name"],
            target_name=obj["name"],
            edge_type=rule.edge_type,
            confidence=round(confidence, 2),
            evidence=evidence[:280],
        )

    # ------------------------------------------------------------------ public
    def extract(self, text: str, mentions: Sequence[Dict]) -> List[ExtractedRelation]:
        """Find relations between mentions, sentence by sentence."""
        relations: List[ExtractedRelation] = []
        seen: Set[Tuple[str, str, str]] = set()

        for sent_start, sentence in self._sentences(text):
            sent_end = sent_start + len(sentence)
            local = self._mentions_in(mentions, sent_start, sent_end)
            if len(local) < 2:
                continue

            for rule in self.rules:
                verb_pattern = rf"\b(?:{rule.verbs})\b"
                if rule.passive:
                    # Require the agentive "by" so we do not treat every past
                    # participle as passive voice.
                    verb_pattern = rf"\b(?:{rule.verbs})\s+(?:by|with)\b"

                for verb_match in re.finditer(verb_pattern, sentence, re.IGNORECASE):
                    verb_start = sent_start + verb_match.start()
                    verb_end = sent_start + verb_match.end()

                    before = [m for m in local if m["end"] <= verb_start]
                    after = [m for m in local if m["start"] >= verb_end]
                    if not before or not after:
                        continue

                    left = before[-1]
                    # Coordination: "directed Interstellar, Dunkirk and Oppenheimer"
                    # should yield an edge to each, not only the first.
                    right_candidates = [after[0]]
                    for candidate in after[1:3]:
                        gap = sentence[
                            candidate["start"] - sent_start - 12 : candidate["start"] - sent_start
                        ]
                        if _COORDINATION.search(gap):
                            right_candidates.append(candidate)
                        else:
                            break

                    for position, right in enumerate(right_candidates):
                        distance = right["start"] - left["end"]
                        penalty = 0.0 if distance <= 80 else 0.06 if distance <= 160 else 0.12
                        penalty += 0.03 * position  # later coordinated items are weaker

                        relation = self._emit(
                            rule, left, right, sentence, distance_penalty=penalty
                        )
                        if relation is None:
                            continue
                        key = (
                            relation.source_name.lower(),
                            relation.edge_type,
                            relation.target_name.lower(),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        relations.append(relation)

        return relations

"""Contextual entity type inference for the regex extraction fallback.

Without a real NER model every entity was labelled with a single default type, so
`Inception` became `canon_person_inception` rather than `canon_film_inception`.
Canonical ids are label-scoped by design (the film "Dune" and the studio "Dune"
are different entities), so a wrong label produces a wrong id, and every
ground-truth lookup misses. It also breaks relation extraction, because
`Person DIRECTED Person` is correctly rejected as a schema violation.

This module recovers the type from the surrounding sentence using linguistic cues
that real prose actually contains:

    "directed by Christopher Nolan"      -> Person  (agent of a directing verb)
    "Inception is a 2010 film"           -> Film    (copular definition)
    "released by OpenAI"                 -> Organization
    "Warner Bros. distributed"           -> Studio  (corporate suffix)

It is a heuristic and not a replacement for GLiNER. It is what makes the regex
path usable when model weights are unavailable, and it degrades to the schema
default rather than guessing wildly.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from app.core.tenant_schema import TenantGraphSchema

# Suffixes that identify an organization regardless of domain.
_ORG_SUFFIXES = re.compile(
    r"\b(?:Inc|Corp|Corporation|Ltd|LLC|GmbH|Bros|Brothers|Studios?|Pictures|"
    r"Entertainment|Company|Labs?|Technologies|AI|Institute|University|"
    r"Foundation|Group|Partners|Media|Networks?)\b\.?",
    re.IGNORECASE,
)

# A capitalized token pair that looks like a personal name.
_PERSON_NAME = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+){1,2}$")

# Cues placed BEFORE the mention, keyed by the label they imply.
_PRE_CUES: Dict[str, List[str]] = {
    "Person": [
        r"directed by", r"written by", r"produced by", r"starring", r"stars",
        r"composed by", r"created by", r"founded by", r"led by", r"authored by",
        r"co-founded by", r"developed by", r"actor", r"actress", r"director",
        r"filmmaker", r"composer", r"screenwriter", r"researcher", r"scientist",
    ],
    "Organization": [
        r"released by", r"developed by", r"published by", r"funded by",
        r"acquired by", r"owned by", r"backed by", r"company", r"startup",
        r"lab(?:oratory)?", r"organization",
    ],
    "Studio": [
        r"distributed by", r"produced by", r"released by", r"studio",
    ],
    "Film": [
        r"film", r"movie", r"feature", r"sequel to", r"prequel to",
        r"adaptation of", r"directed", r"screened",
    ],
    "Model": [
        r"model", r"released", r"trained", r"fine-?tuned", r"LLM",
        r"language model", r"architecture",
    ],
    "Technique": [
        r"technique", r"method", r"algorithm", r"approach", r"mechanism",
        r"using", r"based on", r"uses",
    ],
    "Paper": [r"paper", r"preprint", r"publication", r"article titled"],
    "Dataset": [r"trained on", r"dataset", r"corpus", r"benchmark data"],
    "Benchmark": [r"evaluated on", r"benchmark", r"scored on", r"tested on"],
    "Hardware": [r"runs on", r"GPU", r"TPU", r"accelerator", r"chip"],
    "Genre": [r"genre", r"genres"],
    "Award": [r"award", r"prize", r"Oscar", r"nominated for", r"won the"],
    # Deliberately narrow. Cues like a bare "in" or "from" precede almost any noun
    # phrase and made Country the catch-all label for unrelated entities.
    "Country": [r"set in", r"filmed in", r"born in", r"released in theaters in"],
}

# Cues placed AFTER the mention.
_POST_CUES: Dict[str, List[str]] = {
    "Person": [
        r"directed", r"wrote", r"produced", r"starred", r"composed", r"said",
        r"founded", r"created", r"joined", r"was born", r"plays", r"portrays",
    ],
    "Film": [
        r"was released", r"premiered", r"grossed", r"was filmed", r"is a \d{4}",
        r"received .{0,20}reviews", r"box office",
    ],
    "Organization": [
        r"released", r"announced", r"published", r"develops?", r"acquired",
        r"was founded", r"is a company",
    ],
    "Studio": [r"distributed", r"released the film", r"produced the film"],
    "Model": [
        r"was released", r"outperforms", r"was trained", r"achieves",
        r"is a large language model", r"has \d+ (?:billion|million) parameters",
    ],
    "Technique": [r"is a technique", r"is a method", r"allows", r"enables"],
    "Benchmark": [r"benchmark", r"evaluation suite"],
}

# Copular definitions: "X is a 2010 science fiction film" -> the noun decides.
_COPULAR = re.compile(
    r"\bis\s+(?:a|an|the)\s+(?:[\w\-]+\s+){0,4}?"
    r"(film|movie|director|actor|actress|composer|writer|producer|company|"
    r"organization|studio|model|technique|method|algorithm|architecture|"
    r"dataset|benchmark|paper|genre|award|country|startup|laboratory)\b",
    re.IGNORECASE,
)

_COPULAR_TO_LABEL = {
    "film": "Film", "movie": "Film",
    "director": "Person", "actor": "Person", "actress": "Person",
    "composer": "Person", "writer": "Person", "producer": "Person",
    "company": "Organization", "organization": "Organization",
    "startup": "Organization", "laboratory": "Organization",
    "studio": "Studio",
    "model": "Model", "architecture": "Model",
    "technique": "Technique", "method": "Technique", "algorithm": "Technique",
    "dataset": "Dataset", "benchmark": "Benchmark", "paper": "Paper",
    "genre": "Genre", "award": "Award", "country": "Country",
}


class TypeInferencer:
    """Infers a schema-valid vertex label for a mention from its context."""

    WINDOW = 60  # characters of context inspected on each side

    def __init__(self, schema: TenantGraphSchema) -> None:
        self.schema = schema
        self._valid = schema.vertex_labels

    # ------------------------------------------------------------------ helpers
    def _first_valid(self, *candidates: Optional[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate and candidate in self._valid:
                return candidate
        return None

    def _copular_label(self, text: str, start: int, end: int) -> Optional[str]:
        """`X is a <noun>` is the strongest available signal, so check it first."""
        after = text[end : end + 120]
        match = _COPULAR.search(after)
        if match:
            return _COPULAR_TO_LABEL.get(match.group(1).lower())
        return None

    def _cue_label(self, text: str, start: int, end: int) -> Optional[str]:
        before = text[max(0, start - self.WINDOW) : start].lower()
        after = text[end : end + self.WINDOW].lower()

        scores: Dict[str, int] = {}
        for label, patterns in _PRE_CUES.items():
            if label not in self._valid:
                continue
            for pattern in patterns:
                # Anchor to the end of the preceding window: "directed by X".
                if re.search(pattern + r"[\s,]*$", before):
                    scores[label] = scores.get(label, 0) + 3
                elif re.search(r"\b" + pattern + r"\b", before):
                    scores[label] = scores.get(label, 0) + 1

        for label, patterns in _POST_CUES.items():
            if label not in self._valid:
                continue
            for pattern in patterns:
                if re.match(r"[\s,]*" + pattern, after):
                    scores[label] = scores.get(label, 0) + 3
                elif re.search(r"\b" + pattern + r"\b", after):
                    scores[label] = scores.get(label, 0) + 1

        if not scores:
            return None
        return max(scores.items(), key=lambda kv: kv[1])[0]

    def _shape_label(self, name: str) -> Optional[str]:
        """Fall back to the shape of the name itself."""
        if _ORG_SUFFIXES.search(name):
            return self._first_valid("Studio", "Organization")
        # Model names: GPT-4, BERT, DALL-E 2, LLaMA-2
        if re.match(r"^[A-Z][A-Za-z]*[-\s]?\d+$", name) or re.match(r"^[A-Z]{2,}(?:-\w+)?$", name):
            return self._first_valid("Model", "Technique")
        if _PERSON_NAME.match(name):
            return self._first_valid("Person")
        return None

    def _subject_label(self, name: str, text: str) -> Optional[str]:
        """Type the document's subject from its opening definition.

        Encyclopedic articles define their subject in the first sentence
        ("Inception is a 2010 science fiction film..."). That sentence types the
        entity far more reliably than any later passing mention, and the subject is
        exactly the entity most queries are about — so getting it wrong is costly.
        """
        opening = text[:400]
        match = _COPULAR.search(opening)
        if not match:
            return None
        # Only apply when the mention actually is the subject of that sentence.
        subject_span = opening[: match.start()]
        if name.lower() not in subject_span.lower():
            return None
        return _COPULAR_TO_LABEL.get(match.group(1).lower())

    # ------------------------------------------------------------------ public
    def infer(self, name: str, text: str, start: int, end: int, default: str) -> Tuple[str, float]:
        """Return (label, confidence) for one mention.

        Signals are ordered by reliability: an explicit copular definition beats a
        verb cue, which beats the shape of the name, which beats the default.
        """
        copular = self._copular_label(text, start, end)
        if copular and copular in self._valid:
            return copular, 0.85

        subject = self._subject_label(name, text)
        if subject and subject in self._valid:
            return subject, 0.82

        cue = self._cue_label(text, start, end)
        if cue:
            return cue, 0.72

        shape = self._shape_label(name)
        if shape:
            return shape, 0.65

        return default, 0.45

    def infer_batch(
        self, mentions: Sequence[Dict], text: str, default: str
    ) -> List[Dict]:
        """Infer labels for all mentions, then reconcile duplicates.

        A name mentioned several times in a document should resolve to one type;
        the highest-confidence occurrence wins, so a passing reference does not
        override a definitional one.
        """
        for mention in mentions:
            label, confidence = self.infer(
                mention["name"], text, mention.get("start", 0), mention.get("end", 0), default
            )
            mention["label"] = label
            mention["confidence"] = round(confidence, 2)

        best: Dict[str, Tuple[str, float]] = {}
        for mention in mentions:
            key = mention["name"].lower()
            current = best.get(key)
            if current is None or mention["confidence"] > current[1]:
                best[key] = (mention["label"], mention["confidence"])

        for mention in mentions:
            label, confidence = best[mention["name"].lower()]
            mention["label"] = label
            mention["confidence"] = confidence

        return list(mentions)

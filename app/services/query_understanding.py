"""Query understanding: what is this question, and what does answering it need?

Runs before any retrieval so the router can decide which paths to spend. Three
outputs:

  intent      LOCAL   anchored on a named entity ("what treats cough?")
              GLOBAL  thematic, no entity to anchor on ("what are the main topics?")
              LEXICAL exact-string lookup (codes, identifiers, quoted phrases)
              HYBRID  none of the above dominates; run everything

  mentions    candidate entity surface forms, stopword-filtered
  hops        how far traversal must reach to answer

Intent matters because the paths have very different costs. A global query has no
seed to traverse from, so running graph retrieval on it buys nothing; a lexical
query wants exact matching, which dense vectors are the wrong tool for.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Set

from app.services.extraction_service import normalize_entity_name


class QueryIntent(str, Enum):
    LOCAL = "local"
    GLOBAL = "global"
    LEXICAL = "lexical"
    HYBRID = "hybrid"
    # Underspecified: retrieving anything would present a guess as an answer.
    CLARIFY = "clarify"


# Queries that carry no retrievable content on their own. Anaphora needs an
# antecedent; a bare continuation needs a prior turn. Retrieving arbitrary chunks
# for these looks like an answer while being a coin flip, which is worse than
# admitting the query cannot be resolved.
_ANAPHORIC = frozenset({
    "it", "that", "this", "they", "them", "those", "these", "he", "she",
    "him", "her", "his", "hers", "its", "their", "theirs",
})

_CONTINUATION = frozenset({
    "more", "next", "another", "again", "continue", "else", "other", "others",
    "further", "additional", "rest", "remaining",
})

# Container words that name a *set* rather than a subject. Distinct from broad
# topic words like "treatment" or "symptoms", which still retrieve usefully
# because documents discuss them directly. The test is whether the word could
# plausibly be the topic of a passage: "treatment" can, "everything" cannot.
_BARE_CATEGORY = frozenset({
    "herbs", "herb", "documents", "document", "data", "info", "information",
    "stuff", "things", "items", "entries", "records", "results",
    "everything", "anything", "something",
})

_ANAPHORIC_PHRASE = re.compile(
    r"^\s*(?:what about|how about|and|but|ok|okay|then|so)?\s*"
    r"(?:that|this|it|those|these|them|they)\s*\??\s*$",
    re.IGNORECASE,
)


# Phrasings that ask about a corpus as a whole rather than about any entity.
_GLOBAL_MARKERS = re.compile(
    r"\b(?:overall|in general|generally|main|major|key|common|overview|"
    r"summar(?:y|ise|ize)|themes?|topics?|patterns?|trends?|categories|"
    r"what kinds? of|what types? of|across (?:the|all)|most (?:common|frequent)|"
    r"how many|list all|everything about|broadly)\b",
    re.IGNORECASE,
)

# Phrasings that anchor on a specific named thing.
_LOCAL_MARKERS = re.compile(
    r"\b(?:who|which|what) (?:is|are|was|were|treats?|causes?|directed|released|"
    r"made|wrote|composed|developed|created)\b|"
    r"\b(?:tell me about|explain|describe|details? (?:of|about)|used for|"
    r"related to|connected to|associated with)\b",
    re.IGNORECASE,
)

# Multi-hop phrasing: the answer is reached through an intermediate entity.
_MULTIHOP_MARKERS = re.compile(
    r"\b(?:other|also|both|besides|apart from|as well as|else|"
    r"the (?:director|author|creator|maker|company|organization) of|"
    r"who (?:directed|wrote|created|released) .{0,40}(?:also|other)|"
    r"connect(?:s|ed|ion)|relationship between|link between|"
    r"same .{0,20}as|shared|in common)\b",
    re.IGNORECASE,
)

_QUOTED = re.compile(r'"([^"]{2,80})"|\'([^\']{2,80})\'')

# Identifier-shaped tokens: model numbers, codes, acronyms, versioned names.
_IDENTIFIER = re.compile(r"\b(?:[A-Z]{2,}[-_]?\d*|\w+[-_]\d+|\d+[A-Za-z]+)\b")

_QUERY_STOPWORDS: Set[str] = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "done", "doing", "can", "could", "shall", "should",
    "will", "would", "may", "might", "must", "have", "has", "had",
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "there", "here", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "about", "into", "over", "after", "before", "also",
    "all", "any", "some", "other", "another", "same", "such", "only", "own",
    "me", "my", "you", "your", "them", "they", "their", "it", "its",
    "show", "tell", "give", "list", "find", "get", "make", "made", "used",
    "things", "stuff", "info", "information", "main", "key", "common",
}

_MENTION_RE = re.compile(r"\b[A-Z][a-zA-Z0-9'’\-]*(?:\s+[A-Z][a-zA-Z0-9'’\-]*)*\b")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")


@dataclass
class QueryAnalysis:
    """What the router needs to decide how to spend retrieval."""

    query: str
    intent: QueryIntent
    mentions: List[str] = field(default_factory=list)
    quoted_phrases: List[str] = field(default_factory=list)
    identifiers: List[str] = field(default_factory=list)
    suggested_hops: int = 1
    confidence: float = 0.0
    signals: List[str] = field(default_factory=list)

    # Set when the query cannot be resolved as written.
    needs_clarification: bool = False
    clarification_prompt: str = ""
    resolved_query: str = ""      # rewritten using conversation context
    resolved_from_context: bool = False

    @property
    def has_anchor(self) -> bool:
        """Whether anything in the query can seed a graph traversal."""
        return bool(self.mentions or self.identifiers)

    @property
    def effective_query(self) -> str:
        """The query retrieval should actually run."""
        return self.resolved_query or self.query

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "mentions": self.mentions,
            "quoted_phrases": self.quoted_phrases,
            "identifiers": self.identifiers,
            "suggested_hops": self.suggested_hops,
            "confidence": round(self.confidence, 3),
            "signals": self.signals,
            "needs_clarification": self.needs_clarification,
            "clarification_prompt": self.clarification_prompt,
            "resolved_from_context": self.resolved_from_context,
            "effective_query": self.effective_query,
        }


class QueryUnderstanding:
    """Classifies a query and extracts what retrieval will need from it."""

    MAX_MENTIONS = 8

    def analyze(
        self, query: str, conversation_context: Optional[Sequence[str]] = None
    ) -> QueryAnalysis:
        text = (query or "").strip()
        if not text:
            return QueryAnalysis(
                query="",
                intent=QueryIntent.CLARIFY,
                needs_clarification=True,
                clarification_prompt="Could you tell me what you would like to know?",
            )

        # Underspecified queries are resolved against conversation context when
        # there is any, and otherwise surfaced as a clarification request. The
        # alternative - retrieving something plausible - presents a guess as an
        # answer, which is the worse failure.
        underspecified, reason = self._is_underspecified(text)
        if underspecified:
            resolved = self._resolve_from_context(text, conversation_context)
            if resolved:
                analysis = self.analyze(resolved)
                analysis.query = text
                analysis.resolved_query = resolved
                analysis.resolved_from_context = True
                analysis.signals.append(f"resolved_from_context:{reason}")
                return analysis

            return QueryAnalysis(
                query=text,
                intent=QueryIntent.CLARIFY,
                needs_clarification=True,
                clarification_prompt=self._clarification_prompt(text, reason),
                confidence=0.9,
                signals=[f"underspecified:{reason}"],
            )

        quoted = self._quoted_phrases(text)
        identifiers = self._identifiers(text)
        mentions = self._mentions(text)
        signals: List[str] = []

        global_score = 0.0
        local_score = 0.0
        lexical_score = 0.0

        is_global_phrasing = bool(_GLOBAL_MARKERS.search(text))
        if is_global_phrasing:
            global_score += 3.0
            signals.append("global_phrasing")
        if _LOCAL_MARKERS.search(text):
            local_score += 1.5
            signals.append("local_phrasing")

        if mentions:
            # Capitalized words in a thematic question ("what are the Main Themes")
            # are phrasing, not entities. Counting them as anchors made every
            # global query look local, which is the classification that matters
            # most to get right: it decides whether communities are reachable.
            if is_global_phrasing:
                signals.append("mentions_discounted_global")
            else:
                local_score += 1.0 + min(len(mentions), 3) * 0.3
                signals.append(f"{len(mentions)}_mentions")
        else:
            # No entity to anchor on: traversal has nothing to start from.
            global_score += 1.0
            signals.append("no_anchor")

        if quoted:
            lexical_score += 2.5
            signals.append("quoted_phrase")
        if identifiers:
            lexical_score += 1.5
            signals.append("identifier_token")

        # A short query with a single capitalized token is usually a lookup.
        if len(text.split()) <= 3 and mentions:
            lexical_score += 0.8
            local_score += 0.5
            signals.append("short_lookup")

        hops = 1
        if _MULTIHOP_MARKERS.search(text):
            hops = 2
            local_score += 1.0
            signals.append("multi_hop_phrasing")

        scores = {
            QueryIntent.GLOBAL: global_score,
            QueryIntent.LOCAL: local_score,
            QueryIntent.LEXICAL: lexical_score,
        }
        best_intent, best_score = max(scores.items(), key=lambda kv: kv[1])
        runner_up = sorted(scores.values(), reverse=True)[1]

        # A narrow margin means the signals disagree; run everything rather than
        # commit to a classification the evidence does not support.
        if best_score <= 0 or (best_score - runner_up) < 0.75:
            intent = QueryIntent.HYBRID
            confidence = 0.4
            signals.append("ambiguous")
        else:
            intent = best_intent
            confidence = min(1.0, best_score / 4.0)

        return QueryAnalysis(
            query=text,
            intent=intent,
            mentions=mentions,
            quoted_phrases=quoted,
            identifiers=identifiers,
            suggested_hops=hops,
            confidence=confidence,
            signals=signals,
        )

    # --------------------------------------------------------- underspecified
    @staticmethod
    def _is_underspecified(text: str) -> tuple[bool, str]:
        """Whether the query names nothing retrievable on its own.

        Deliberately narrow. A bare topic word like "treatment" still retrieves
        usefully, so only queries that are genuinely unresolvable qualify:
        anaphora with no antecedent, continuations with nothing to continue, and
        category words naming no entity.
        """
        stripped = text.strip().strip("?!.").lower()
        words = stripped.split()

        if _ANAPHORIC_PHRASE.match(text):
            return True, "anaphora_without_antecedent"

        if len(words) == 1:
            word = words[0]
            if word in _ANAPHORIC:
                return True, "anaphora_without_antecedent"
            if word in _CONTINUATION:
                return True, "continuation_without_prior_turn"
            if word in _BARE_CATEGORY:
                return True, "category_without_entity"

        # Two-word forms of the same problem: "more herbs", "other treatments".
        if len(words) == 2 and words[0] in _CONTINUATION and words[1] in _BARE_CATEGORY:
            return True, "category_without_entity"

        return False, ""

    @staticmethod
    def _resolve_from_context(
        text: str, context: Optional[Sequence[str]]
    ) -> Optional[str]:
        """Rewrite an underspecified query using the previous turn.

        Only the most recent turn is used: an anaphor refers to what was just
        said, and reaching further back produces confident nonsense.
        """
        if not context:
            return None

        previous = str(context[-1]).strip()
        if not previous:
            return None

        stripped = text.strip().strip("?!.").lower()
        words = stripped.split()

        # A continuation inherits the previous question wholesale.
        if words and words[0] in _CONTINUATION:
            return previous

        # An anaphor is replaced by the previous turn's subject matter.
        if _ANAPHORIC_PHRASE.match(text) or (words and words[0] in _ANAPHORIC):
            return previous

        return f"{previous} {text}".strip()

    @staticmethod
    def _clarification_prompt(text: str, reason: str) -> str:
        """A question that tells the user what is missing and what to supply."""
        prompts = {
            "anaphora_without_antecedent": (
                f"I'm not sure what \"{text.strip()}\" refers to. "
                "Could you name the topic you'd like to know about?"
            ),
            "continuation_without_prior_turn": (
                "There's no previous question for me to continue from. "
                "What would you like to know more about?"
            ),
            "category_without_entity": (
                f"\"{text.strip()}\" covers a lot of ground. "
                "Could you narrow it down — a specific item, or what you want to know about it?"
            ),
        }
        return prompts.get(
            reason, "Could you rephrase that with a bit more detail?"
        )

    # ------------------------------------------------------------------ parts
    @staticmethod
    def _quoted_phrases(text: str) -> List[str]:
        return [
            (m.group(1) or m.group(2)).strip()
            for m in _QUOTED.finditer(text)
            if (m.group(1) or m.group(2))
        ]

    @staticmethod
    def _identifiers(text: str) -> List[str]:
        found: List[str] = []
        for match in _IDENTIFIER.finditer(text):
            token = match.group(0)
            if token.lower() in _QUERY_STOPWORDS or len(token) < 2:
                continue
            if token not in found:
                found.append(token)
        return found[:6]

    def _mentions(self, text: str) -> List[str]:
        """Candidate entity surface forms, interrogatives stripped."""
        mentions: List[str] = []
        seen: Set[str] = set()

        for match in _MENTION_RE.finditer(text):
            phrase = match.group(0).strip()
            words = phrase.split()
            while words and words[0].lower() in _QUERY_STOPWORDS:
                words.pop(0)
            while words and words[-1].lower() in _QUERY_STOPWORDS:
                words.pop()
            if not words:
                continue
            cleaned = " ".join(words)
            key = normalize_entity_name(cleaned)
            if key and key not in seen and len(key) > 1:
                seen.add(key)
                mentions.append(cleaned)

        # All-lowercase queries still need seeds, so fall back to content words.
        if not mentions:
            for word in _WORD_RE.findall(text):
                lowered = word.lower()
                if lowered in _QUERY_STOPWORDS or len(lowered) < 3:
                    continue
                key = normalize_entity_name(word)
                if key and key not in seen:
                    seen.add(key)
                    mentions.append(word)

        return mentions[: self.MAX_MENTIONS]


query_understanding = QueryUnderstanding()

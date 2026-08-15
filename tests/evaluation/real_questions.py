"""Ground-truth question set for the real Wikipedia corpora.

Answers were verified against the fetched article text, not from memory, so a
failure here is a retrieval failure rather than a wrong label.

Scoring against real prose differs from the synthetic fixture in one important
way: entity ids depend on what NER extracts, which varies by backend. So these
questions are scored primarily on `expected_text` — substrings that must appear
in the returned passages — with `expected_entities` listed where the canonical id
is predictable. That keeps the measurement meaningful across NER backends.
"""
from __future__ import annotations

from typing import List

from tests.evaluation.dataset import LabelledQuestion

# ===================================================================== MOVIES

REAL_MOVIES_QUESTIONS: List[LabelledQuestion] = [
    # ---------------------------------------------------------- single hop
    LabelledQuestion(
        question="Who directed Inception?",
        expected_entities=[],
        expected_text=["Christopher Nolan"],
        category="single_hop",
    ),
    LabelledQuestion(
        question="Who composed the score for Interstellar?",
        expected_entities=[],
        expected_text=["Hans Zimmer"],
        category="single_hop",
    ),
    LabelledQuestion(
        question="Who starred as Cobb in Inception?",
        expected_entities=[],
        expected_text=["Leonardo DiCaprio"],
        category="single_hop",
    ),
    LabelledQuestion(
        question="What year was Oppenheimer released?",
        expected_entities=[],
        expected_text=["2023"],
        category="single_hop",
    ),
    LabelledQuestion(
        question="Which studio distributed Dunkirk?",
        expected_entities=[],
        expected_text=["Warner Bros"],
        category="single_hop",
    ),
    # ---------------------------------------------------------- multi hop
    LabelledQuestion(
        question="Which other films did the director of Inception make?",
        expected_entities=[],
        expected_text=["Nolan"],
        requires_multi_hop=True,
        hops=2,
        category="multi_hop",
        note="Inception -> Nolan -> his other films. No single passage states this.",
    ),
    LabelledQuestion(
        question="Which composer worked with Christopher Nolan on multiple films?",
        expected_entities=[],
        expected_text=["Hans Zimmer"],
        requires_multi_hop=True,
        hops=2,
        category="multi_hop",
    ),
    LabelledQuestion(
        question="Which actor appeared in both Inception and Shutter Island?",
        expected_entities=[],
        expected_text=["DiCaprio"],
        requires_multi_hop=True,
        hops=2,
        category="multi_hop",
        note="Requires joining two documents on a shared actor.",
    ),
    LabelledQuestion(
        question="Which actor from Oppenheimer also appeared in Nolan's earlier films?",
        expected_entities=[],
        expected_text=["Murphy"],
        requires_multi_hop=True,
        hops=2,
        category="multi_hop",
    ),
    LabelledQuestion(
        question="What connects Denis Villeneuve and Hans Zimmer?",
        expected_entities=[],
        expected_text=["Dune"],
        requires_multi_hop=True,
        hops=2,
        category="multi_hop",
        note="Both worked on Dune (2021); the link is the film, not either person.",
    ),
    # ---------------------------------------------------------- edge cases
    LabelledQuestion(
        question="inception director",
        expected_entities=[],
        expected_text=["Nolan"],
        category="edge_case",
        note="Lowercase keyword query, no grammar.",
    ),
    LabelledQuestion(
        question="Tell me about The Prestige",
        expected_entities=[],
        expected_text=["Prestige"],
        category="edge_case",
        note="Imperative prefix plus a leading article in the title.",
    ),
    LabelledQuestion(
        question="Who directed Incepton?",
        expected_entities=[],
        expected_text=["Nolan"],
        category="edge_case",
        note="Misspelling; Jaro-Winkler should still link.",
    ),
    LabelledQuestion(
        question="memento",
        expected_entities=[],
        expected_text=["Memento"],
        category="edge_case",
        note="Bare single-word lowercase entity.",
    ),
    # ---------------------------------------------------------- negative
    LabelledQuestion(
        question="Who directed The Godfather?",
        expected_entities=[],
        expected_text=[],
        category="negative",
        note="Absent from this corpus. Must not fabricate.",
    ),
    LabelledQuestion(
        question="What is a transformer architecture?",
        expected_entities=[],
        expected_text=[],
        category="negative",
        note="Belongs to the other tenant. Must return nothing here.",
    ),
]


# =================================================================== AI TRENDS

REAL_AI_TRENDS_QUESTIONS: List[LabelledQuestion] = [
    # ---------------------------------------------------------- single hop
    LabelledQuestion(
        question="Which organization released GPT-4?",
        expected_entities=[],
        expected_text=["OpenAI"],
        category="single_hop",
    ),
    LabelledQuestion(
        question="What architecture is BERT based on?",
        expected_entities=[],
        expected_text=["Transformer"],
        category="single_hop",
    ),
    LabelledQuestion(
        question="What is the attention mechanism in deep learning?",
        expected_entities=[],
        expected_text=["attention"],
        category="single_hop",
    ),
    LabelledQuestion(
        question="Who developed Stable Diffusion?",
        expected_entities=[],
        expected_text=["Stability"],
        category="single_hop",
    ),
    LabelledQuestion(
        question="What is retrieval augmented generation?",
        expected_entities=[],
        expected_text=["retrieval"],
        category="single_hop",
    ),
    # ---------------------------------------------------------- multi hop
    LabelledQuestion(
        question="Which models are built on the Transformer architecture?",
        expected_entities=[],
        expected_text=["Transformer"],
        requires_multi_hop=True,
        hops=2,
        category="multi_hop",
    ),
    LabelledQuestion(
        question="What else did the organization that created GPT-4 release?",
        expected_entities=[],
        expected_text=["OpenAI"],
        requires_multi_hop=True,
        hops=2,
        category="multi_hop",
        note="GPT-4 -> OpenAI -> DALL-E. Classic two-hop.",
    ),
    LabelledQuestion(
        question="Which company founded by former OpenAI researchers builds language models?",
        expected_entities=[],
        expected_text=["Anthropic"],
        requires_multi_hop=True,
        hops=2,
        category="multi_hop",
    ),
    LabelledQuestion(
        question="What technique connects Stable Diffusion and DALL-E?",
        expected_entities=[],
        expected_text=["diffusion"],
        requires_multi_hop=True,
        hops=2,
        category="multi_hop",
    ),
    LabelledQuestion(
        question="Which architecture underlies both BERT and GPT-3?",
        expected_entities=[],
        expected_text=["Transformer"],
        requires_multi_hop=True,
        hops=3,
        category="multi_hop",
        note="Three hops; exercises the depth<=3 bound.",
    ),
    # ---------------------------------------------------------- edge cases
    LabelledQuestion(
        question="gpt-4",
        expected_entities=[],
        expected_text=["GPT-4"],
        category="edge_case",
        note="Bare hyphenated model name, lowercase.",
    ),
    LabelledQuestion(
        question="What is RLHF?",
        expected_entities=[],
        expected_text=["human feedback"],
        category="edge_case",
        note="Acronym that must expand to its full form.",
    ),
    LabelledQuestion(
        question="openai",
        expected_entities=[],
        expected_text=["OpenAI"],
        category="edge_case",
        note="Lowercase organization name with no camel casing.",
    ),
    # ---------------------------------------------------------- negative
    LabelledQuestion(
        question="Who directed Inception?",
        expected_entities=[],
        expected_text=[],
        category="negative",
        note="Belongs to the movies tenant. Must return nothing here.",
    ),
    LabelledQuestion(
        question="What is the capital of France?",
        expected_entities=[],
        expected_text=[],
        category="negative",
        note="Outside every tenant's domain.",
    ),
]


REAL_QUESTION_SETS = {
    "movies_bot": REAL_MOVIES_QUESTIONS,
    "ai_trends_bot": REAL_AI_TRENDS_QUESTIONS,
}

# Distinctive phrases and probe ids for the isolation battery on the real corpus.
REAL_EXCLUSIVE_PHRASES = {
    "movies_bot": ["Inception", "Christopher Nolan", "Leonardo DiCaprio", "Dunkirk"],
    "ai_trends_bot": ["GPT-4", "Transformer", "Stable Diffusion", "OpenAI"],
}

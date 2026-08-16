"""TMDB evaluation set: a corpus where the graph should actually matter.

The Ayurveda ablation returned graph lift of +0.020, with zero queries answered
only by traversal. That was a property of the data, not a defect: each row is
self-contained, so every fact about a disease sits in that disease's own chunk
and vector search reaches it directly. There was nothing to traverse.

TMDB has the opposite shape. Within the ingested slice, 941 directors and 4,995
actors appear in more than one film, so a question like "which other films did
the director of The Dark Knight Rises make?" has an answer that exists in no
single record: film one names Nolan, and Nolan's other films are named in
different records entirely. That is precisely the structure vector search cannot
follow and traversal can.

Every question below was generated from verified relationships in the data rather
than written from memory, so each answer is checkable against the source.
"""
from __future__ import annotations

from typing import List

from tests.evaluation.edge_case_suite import Category, EdgeCaseQuery, PassRule

TMDB = "tmdb_films"


# ===================================================== SINGLE HOP (answer in one record)
TMDB_SINGLE_HOP: List[EdgeCaseQuery] = [
    EdgeCaseQuery("Who directed Avatar?", Category.SEMANTIC, TMDB,
                  PassRule.RETRIEVES_TEXT, expect_text=["cameron"],
                  relevant_chunk_markers=["Avatar"]),
    EdgeCaseQuery("Who directed Spectre?", Category.SEMANTIC, TMDB,
                  PassRule.RETRIEVES_TEXT, expect_text=["mendes"],
                  relevant_chunk_markers=["Spectre"]),
    EdgeCaseQuery("Who directed The Dark Knight Rises?", Category.SEMANTIC, TMDB,
                  PassRule.RETRIEVES_TEXT, expect_text=["nolan"],
                  relevant_chunk_markers=["The Dark Knight Rises"]),
    EdgeCaseQuery("Who directed John Carter?", Category.SEMANTIC, TMDB,
                  PassRule.RETRIEVES_TEXT, expect_text=["stanton"],
                  relevant_chunk_markers=["John Carter"]),
    EdgeCaseQuery("Who directed Spider-Man 3?", Category.SEMANTIC, TMDB,
                  PassRule.RETRIEVES_TEXT, expect_text=["raimi"],
                  relevant_chunk_markers=["Spider-Man 3"]),
    EdgeCaseQuery("What genre is Avatar?", Category.SEMANTIC, TMDB,
                  PassRule.RETRIEVES_TEXT, expect_text=["avatar"],
                  relevant_chunk_markers=["Avatar"]),
    EdgeCaseQuery("Who starred in Titanic?", Category.SEMANTIC, TMDB,
                  PassRule.RETRIEVES_TEXT, expect_text=["titanic"],
                  relevant_chunk_markers=["Titanic"]),
    EdgeCaseQuery("Which company produced The Avengers?", Category.SEMANTIC, TMDB,
                  PassRule.RETRIEVES_TEXT, expect_text=["avengers"],
                  relevant_chunk_markers=["The Avengers"]),
]


# ============================ MULTI-HOP (answer spans records; the point of the graph)
#
# Each of these requires two steps: film -> person -> that person's other films.
# The second film is never named in the first film's record, so a retriever that
# only ranks documents by similarity to the query cannot reach it.
TMDB_MULTI_HOP: List[EdgeCaseQuery] = [
    EdgeCaseQuery(
        "Which other films did the director of The Dark Knight Rises make?",
        Category.MULTI_HOP, TMDB, PassRule.RETRIEVES_TEXT,
        expect_text=["nolan"],
        relevant_chunk_markers=["The Dark Knight", "Interstellar"],
        requires_multi_hop=True, hops=2,
        note="Nolan -> The Dark Knight, Interstellar. Neither is named in the "
             "Dark Knight Rises record."),
    EdgeCaseQuery(
        "Which other films did the director of Robin Hood make?",
        Category.MULTI_HOP, TMDB, PassRule.RETRIEVES_TEXT,
        expect_text=["scott"],
        relevant_chunk_markers=["Prometheus", "Exodus"],
        requires_multi_hop=True, hops=2,
        note="Ridley Scott -> Prometheus, Exodus"),
    EdgeCaseQuery(
        "Which other films did the director of Transformers: Revenge of the Fallen make?",
        Category.MULTI_HOP, TMDB, PassRule.RETRIEVES_TEXT,
        expect_text=["bay"],
        relevant_chunk_markers=["Transformers: Age of Extinction"],
        requires_multi_hop=True, hops=2,
        note="Michael Bay -> other Transformers films"),
    EdgeCaseQuery(
        "Which other films did the director of Superman Returns make?",
        Category.MULTI_HOP, TMDB, PassRule.RETRIEVES_TEXT,
        expect_text=["singer"],
        relevant_chunk_markers=["X-Men: Days of Future Past"],
        requires_multi_hop=True, hops=2,
        note="Bryan Singer -> X-Men films"),
    EdgeCaseQuery(
        "Which other films did the director of Indiana Jones and the Kingdom of the "
        "Crystal Skull make?",
        Category.MULTI_HOP, TMDB, PassRule.RETRIEVES_TEXT,
        expect_text=["spielberg"],
        relevant_chunk_markers=["War of the Worlds"],
        requires_multi_hop=True, hops=2,
        note="Spielberg -> War of the Worlds, The BFG"),
    EdgeCaseQuery(
        "Which other films did the director of The Hobbit: The Battle of the Five "
        "Armies make?",
        Category.MULTI_HOP, TMDB, PassRule.RETRIEVES_TEXT,
        expect_text=["jackson"],
        relevant_chunk_markers=["King Kong"],
        requires_multi_hop=True, hops=2,
        note="Peter Jackson -> King Kong, Desolation of Smaug"),
    EdgeCaseQuery(
        "Which other films did the director of 2012 make?",
        Category.MULTI_HOP, TMDB, PassRule.RETRIEVES_TEXT,
        expect_text=["emmerich"],
        relevant_chunk_markers=["White House Down"],
        requires_multi_hop=True, hops=2,
        note="Roland Emmerich -> White House Down, Independence Day"),
    EdgeCaseQuery(
        "Which other films did the director of Pirates of the Caribbean: At World's "
        "End make?",
        Category.MULTI_HOP, TMDB, PassRule.RETRIEVES_TEXT,
        expect_text=["verbinski"],
        relevant_chunk_markers=["The Lone Ranger"],
        requires_multi_hop=True, hops=2,
        note="Gore Verbinski -> The Lone Ranger"),
]


# ============================ SHARED-ENTITY (two films joined by a person)
TMDB_SHARED_ENTITY: List[EdgeCaseQuery] = [
    EdgeCaseQuery(
        "Which actor appears in both Edge of Tomorrow and Mission: Impossible - "
        "Rogue Nation?",
        Category.MULTI_HOP, TMDB, PassRule.RETRIEVES_TEXT,
        expect_text=["cruise"],
        relevant_chunk_markers=["Edge of Tomorrow"],
        requires_multi_hop=True, hops=2),
    EdgeCaseQuery(
        "Which actor appears in both X-Men: The Last Stand and X-Men: Days of "
        "Future Past?",
        Category.MULTI_HOP, TMDB, PassRule.RETRIEVES_TEXT,
        expect_text=["jackman"],
        relevant_chunk_markers=["X-Men: Days of Future Past"],
        requires_multi_hop=True, hops=2),
    EdgeCaseQuery(
        "Which actor appears in both Men in Black 3 and Wild Wild West?",
        Category.MULTI_HOP, TMDB, PassRule.RETRIEVES_TEXT,
        expect_text=["smith"],
        relevant_chunk_markers=["Men in Black 3"],
        requires_multi_hop=True, hops=2),
    EdgeCaseQuery(
        "Which actor appears in both Terminator Genisys and Terminator 3: Rise of "
        "the Machines?",
        Category.MULTI_HOP, TMDB, PassRule.RETRIEVES_TEXT,
        expect_text=["schwarzenegger"],
        relevant_chunk_markers=["Terminator Genisys"],
        requires_multi_hop=True, hops=2),
    EdgeCaseQuery(
        "What connects Maleficent and Alexander?",
        Category.MULTI_HOP, TMDB, PassRule.RETRIEVES_TEXT,
        expect_text=["jolie"],
        relevant_chunk_markers=["Maleficent"],
        requires_multi_hop=True, hops=2),
    EdgeCaseQuery(
        "Which films did Warner Bros. produce?",
        Category.RELATIONSHIP, TMDB, PassRule.RETRIEVES_GRAPH,
        min_graph_edges=1,
        relevant_chunk_markers=["Warner Bros"]),
]


# ============================ RELATIONSHIP (graph edges expected)
TMDB_RELATIONSHIP: List[EdgeCaseQuery] = [
    EdgeCaseQuery("What is Christopher Nolan connected to?", Category.RELATIONSHIP,
                  TMDB, PassRule.RETRIEVES_GRAPH, min_graph_edges=1,
                  relevant_chunk_markers=["The Dark Knight Rises"]),
    EdgeCaseQuery("What is associated with James Cameron?", Category.RELATIONSHIP,
                  TMDB, PassRule.RETRIEVES_GRAPH, min_graph_edges=1,
                  relevant_chunk_markers=["Avatar"]),
    EdgeCaseQuery("Which films relate to Steven Spielberg?", Category.RELATIONSHIP,
                  TMDB, PassRule.RETRIEVES_GRAPH, min_graph_edges=1,
                  relevant_chunk_markers=["War of the Worlds"]),
    EdgeCaseQuery("What is Ridley Scott connected to?", Category.RELATIONSHIP,
                  TMDB, PassRule.RETRIEVES_GRAPH, min_graph_edges=1,
                  relevant_chunk_markers=["Prometheus"]),
]


# ============================ COMPARISON
TMDB_COMPARISON: List[EdgeCaseQuery] = [
    EdgeCaseQuery("Compare Avatar and Titanic", Category.COMPARISON, TMDB,
                  PassRule.RETRIEVES_ANY,
                  relevant_chunk_markers=["Avatar"]),
    EdgeCaseQuery("How do The Dark Knight and The Dark Knight Rises differ?",
                  Category.COMPARISON, TMDB, PassRule.RETRIEVES_ANY,
                  relevant_chunk_markers=["The Dark Knight"]),
    EdgeCaseQuery("What do Prometheus and Alien have in common?",
                  Category.COMPARISON, TMDB, PassRule.RETRIEVES_ANY,
                  relevant_chunk_markers=["Prometheus"]),
]


# ============================ EXACT ENTITY
TMDB_EXACT: List[EdgeCaseQuery] = [
    EdgeCaseQuery("Interstellar", Category.EXACT, TMDB, PassRule.RETRIEVES_TEXT,
                  expect_text=["interstellar"],
                  relevant_chunk_markers=["Interstellar"]),
    EdgeCaseQuery("Prometheus", Category.EXACT, TMDB, PassRule.RETRIEVES_TEXT,
                  expect_text=["prometheus"],
                  relevant_chunk_markers=["Prometheus"]),
    EdgeCaseQuery('"Christopher Nolan"', Category.EXACT, TMDB,
                  PassRule.RETRIEVES_TEXT, expect_text=["nolan"],
                  relevant_chunk_markers=["Nolan"]),
]


TMDB_QUERIES: List[EdgeCaseQuery] = (
    TMDB_SINGLE_HOP + TMDB_MULTI_HOP + TMDB_SHARED_ENTITY
    + TMDB_RELATIONSHIP + TMDB_COMPARISON + TMDB_EXACT
)

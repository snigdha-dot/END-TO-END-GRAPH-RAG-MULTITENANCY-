"""Ingest TMDB films as a corpus with genuine cross-document structure.

TMDB stores cast, crew, genres, and companies as JSON arrays inside each film's
row. Left as raw JSON they are unreadable to both the embedder and the extractor,
so each film is rendered as prose naming its people and companies explicitly.

That rendering is what creates the graph: writing "Directed by: Christopher
Nolan" in two different films' text is what lets extraction produce one canonical
Nolan entity linked to both, and that shared entity is the bridge a multi-hop
traversal walks. A flat record with the same facts buried in JSON produces no
such bridge.

    python -m tests.evaluation.ingest_tmdb --films 300
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from app.core.tenant_context import TenantContext, tenant_scope
from app.services.arcadedb_client import arcadedb_client
from app.services.graph_schema_service import graph_schema_service
from app.services.ingestion_pipeline import ingestion_pipeline

BASE = Path(
    r"C:\Users\snigd\.cache\kagglehub\datasets\tmdb\tmdb-movie-metadata\versions\2"
)
TENANT = "tmdb_films"


def _names(raw: str, limit: int = 8) -> List[str]:
    try:
        return [item["name"] for item in json.loads(raw or "[]")][:limit]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def _directors(raw_crew: str) -> List[str]:
    try:
        return [
            person["name"]
            for person in json.loads(raw_crew or "[]")
            if person.get("job") == "Director"
        ]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def _cast(raw_cast: str, limit: int = 6) -> List[str]:
    try:
        return [person["name"] for person in json.loads(raw_cast or "[]")][:limit]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def render_film(movie: Dict[str, Any], credits: Dict[str, Any]) -> str:
    """Render one film as prose that names its people and companies.

    Sentences are written so relation extraction can find subject-verb-object
    pairs: "Christopher Nolan directed The Dark Knight Rises" produces a DIRECTED
    edge, whereas a JSON blob produces nothing.
    """
    title = movie["title"]
    lines = [f"# {title}", ""]

    year = (movie.get("release_date") or "")[:4]
    if year:
        lines.append(f"{title} is a film released in {year}.")

    directors = _directors(credits.get("crew", ""))
    for director in directors:
        lines.append(f"{director} directed {title}.")

    cast = _cast(credits.get("cast", ""))
    for actor in cast:
        lines.append(f"{actor} starred in {title}.")

    companies = _names(movie.get("production_companies", ""), limit=4)
    for company in companies:
        lines.append(f"{company} produced {title}.")

    genres = _names(movie.get("genres", ""), limit=4)
    if genres:
        lines.append(f"{title} has genre {', '.join(genres)}.")

    overview = (movie.get("overview") or "").strip()
    if overview:
        lines.extend(["", "## Overview", "", overview])

    tagline = (movie.get("tagline") or "").strip()
    if tagline:
        lines.append(f'Tagline: "{tagline}"')

    runtime = movie.get("runtime")
    rating = movie.get("vote_average")
    if runtime or rating:
        lines.append("")
        lines.append("## Details")
        lines.append("")
        if runtime:
            lines.append(f"Runtime: {runtime} minutes.")
        if rating:
            lines.append(f"Average rating: {rating}.")

    return "\n".join(lines)


def load_films(limit: int) -> List[Dict[str, str]]:
    csv.field_size_limit(2_147_483_647)

    credits_by_title: Dict[str, Dict[str, Any]] = {}
    with (BASE / "tmdb_5000_credits.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            credits_by_title[row["title"]] = row

    documents: List[Dict[str, str]] = []
    with (BASE / "tmdb_5000_movies.csv").open(encoding="utf-8") as handle:
        for index, movie in enumerate(csv.DictReader(handle)):
            if index >= limit:
                break
            title = movie.get("title", "")
            if not title:
                continue
            text = render_film(movie, credits_by_title.get(title, {}))
            if len(text) < 100:
                continue
            slug = "".join(
                c if c.isalnum() else "_" for c in title.lower()
            ).strip("_")[:60]
            documents.append({"doc_id": f"film_{slug}", "text": text})

    return documents


async def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest TMDB films")
    parser.add_argument("--films", type=int, default=300)
    parser.add_argument("--batch", type=int, default=25)
    args = parser.parse_args()

    documents = load_films(args.films)
    print(f"rendered {len(documents)} films")

    await arcadedb_client.start()
    if not await arcadedb_client.is_ready():
        print("ArcadeDB is not reachable.")
        await arcadedb_client.close()
        return 2

    await graph_schema_service.provision_tenant(TENANT)
    ctx = TenantContext(tenant_id=TENANT, api_key_id="tmdb", request_id="tmdb")

    totals = {"chunks": 0, "entities": 0, "relationships": 0}
    started = time.perf_counter()

    for index, document in enumerate(documents, start=1):
        with tenant_scope(ctx):
            result = await ingestion_pipeline.ingest_text(
                ctx=ctx,
                doc_id=document["doc_id"],
                content=document["text"],
                # Communities are built once at the end rather than per film:
                # detection over a partial graph would cluster what happens to
                # have been ingested so far.
                build_communities=False,
            )
        totals["chunks"] += result.get("chunks_created", 0)
        totals["entities"] += result.get("entities_written", 0)
        totals["relationships"] += result.get("relationships_created", 0)

        if index % args.batch == 0 or index == len(documents):
            elapsed = time.perf_counter() - started
            print(
                f"  {index}/{len(documents)} films  "
                f"chunks={totals['chunks']} entities={totals['entities']} "
                f"rels={totals['relationships']}  ({elapsed:.0f}s)"
            )

    print()
    print(f"DONE in {time.perf_counter() - started:.0f}s")
    print(f"  chunks       : {totals['chunks']}")
    print(f"  entities     : {totals['entities']}")
    print(f"  relationships: {totals['relationships']}")

    await arcadedb_client.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

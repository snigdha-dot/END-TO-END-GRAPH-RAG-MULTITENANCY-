"""Real-corpus loaders for evaluation.

The hand-written fixtures in `dataset.py` are clean `Subject verb Object` prose,
which flatters a regex extractor and tells you little about production behaviour.
These loaders pull genuine documents — real sentence structure, parenthetical
asides, dates, nested clauses, inconsistent naming — which is what actually
exercises chunking, NER, and entity resolution.

Sources, in order of preference:

  kaggle    A CSV from a Kaggle dataset. Requires ~/.kaggle/kaggle.json. Point it
            at any text-bearing CSV via --kaggle-dataset / --text-column.
  wikipedia Live article extracts via the public REST API. No credentials needed.
  local     Any directory of .txt / .md files you already have.

All three produce the same shape, so the evaluation harness does not care which
one supplied the corpus.
"""
from __future__ import annotations

import csv
import json
import logging
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

_USER_AGENT = "TeamB-GraphRAG-Evaluation/1.0 (retrieval quality benchmark)"


@dataclass
class LoadedCorpus:
    """A set of documents plus provenance, ready for ingestion."""

    tenant_id: str
    source: str
    documents: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def total_chars(self) -> int:
        return sum(len(d) for d in self.documents.values())

    def summary(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "source": self.source,
            "documents": self.document_count,
            "total_chars": self.total_chars,
            "mean_doc_chars": (
                round(self.total_chars / self.document_count) if self.document_count else 0
            ),
        }


# --------------------------------------------------------------------- Wikipedia

# Curated article sets matching the two tenant domains. Deliberately disjoint, so
# a cross-tenant hit remains unambiguous evidence of a leak.
WIKIPEDIA_ARTICLE_SETS: Dict[str, List[str]] = {
    "movies_bot": [
        "Inception", "Interstellar_(film)", "Dunkirk_(2017_film)",
        "Oppenheimer_(film)", "The_Prestige_(film)", "Memento_(film)",
        "Christopher_Nolan", "Leonardo_DiCaprio", "Cillian_Murphy",
        "Michael_Caine", "Hans_Zimmer",
        "The_Departed", "Shutter_Island_(film)", "Martin_Scorsese",
        "Dune_(2021_film)", "Arrival_(film)", "Denis_Villeneuve",
        "Blade_Runner_2049", "Warner_Bros.", "Legendary_Entertainment",
    ],
    "ai_trends_bot": [
        "Transformer_(deep_learning_architecture)", "Attention_(machine_learning)",
        "BERT_(language_model)", "GPT-3", "GPT-4", "Large_language_model",
        "OpenAI", "DeepMind", "Anthropic", "Hugging_Face",
        "Diffusion_model", "Stable_Diffusion", "DALL-E",
        "Retrieval-augmented_generation", "Vector_database",
        "Nearest_neighbor_search", "Word_embedding",
        "Reinforcement_learning_from_human_feedback", "Machine_learning",
        "Deep_learning",
    ],
}


def _http_get(url: str, timeout: int = 30, retries: int = 4) -> bytes:
    """GET with exponential backoff. Wikipedia returns 429 on burst requests."""
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 429 and attempt < retries - 1:
                time.sleep(2.0 * (2**attempt))
                continue
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"Request failed after {retries} attempts: {last_exc}")


def _clean_wikipedia_text(text: str) -> str:
    """Strip artefacts that would otherwise become spurious entities."""
    # Drop reference markers, edit links, and the trailing apparatus sections.
    text = re.sub(r"\[\d+\]|\[edit\]|\[citation needed\]", "", text)
    text = re.split(
        r"\n\s*==\s*(?:See also|References|External links|Further reading|Notes|Bibliography)\s*==",
        text,
        flags=re.IGNORECASE,
    )[0]
    # Normalise heading markup to markdown so the chunker can see structure.
    text = re.sub(r"^===\s*(.+?)\s*===$", r"### \1", text, flags=re.MULTILINE)
    text = re.sub(r"^==\s*(.+?)\s*==$", r"## \1", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_wikipedia(
    tenant_id: str,
    titles: Optional[Sequence[str]] = None,
    max_articles: Optional[int] = None,
    delay_seconds: float = 1.0,
) -> LoadedCorpus:
    """Fetch article extracts from the Wikipedia REST API. No credentials needed."""
    article_titles = list(titles or WIKIPEDIA_ARTICLE_SETS.get(tenant_id, []))
    if max_articles:
        article_titles = article_titles[:max_articles]

    corpus = LoadedCorpus(tenant_id=tenant_id, source="wikipedia")

    for title in article_titles:
        encoded = urllib.parse.quote(title, safe="")
        url = (
            "https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
            f"&explaintext=1&format=json&titles={encoded}&redirects=1"
        )
        try:
            payload = json.loads(_http_get(url).decode("utf-8"))
            pages = payload.get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "")
                if not extract or len(extract) < 500:
                    logger.warning("Skipping '%s': extract too short or missing.", title)
                    continue
                doc_id = f"wiki_{re.sub(r'[^a-zA-Z0-9]+', '_', title).strip('_').lower()}"
                cleaned = _clean_wikipedia_text(extract)
                corpus.documents[doc_id] = cleaned
                corpus.metadata[doc_id] = {
                    "source": "wikipedia",
                    "title": page.get("title", title),
                    "url": f"https://en.wikipedia.org/wiki/{encoded}",
                    "chars": len(cleaned),
                }
        except Exception as exc:  # noqa: BLE001 - one bad article must not abort the load
            logger.warning("Failed to fetch '%s': %s", title, exc)
        time.sleep(delay_seconds)  # courtesy rate limit

    return corpus


# ------------------------------------------------------------------------ Kaggle

def load_kaggle(
    tenant_id: str,
    dataset: str,
    text_column: str,
    title_column: Optional[str] = None,
    max_rows: int = 200,
    min_chars: int = 400,
    download_dir: Optional[Path] = None,
) -> LoadedCorpus:
    """Download a Kaggle dataset and build a corpus from one text column.

    Requires the `kaggle` package and credentials at `~/.kaggle/kaggle.json`.
    Raises with an actionable message when either is missing, rather than silently
    producing an empty corpus.
    """
    creds = Path.home() / ".kaggle" / "kaggle.json"
    if not creds.exists():
        raise RuntimeError(
            "Kaggle credentials not found at ~/.kaggle/kaggle.json.\n"
            "Create an API token at https://www.kaggle.com/settings/account "
            "('Create New Token'), save the file there, then re-run.\n"
            "Alternatively use --source wikipedia, which needs no credentials."
        )

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "The 'kaggle' package is not installed. Run: pip install kaggle"
        ) from exc

    target = download_dir or Path("data/kaggle") / tenant_id
    target.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    logger.info("Downloading Kaggle dataset '%s'...", dataset)
    api.dataset_download_files(dataset, path=str(target), unzip=True, quiet=False)

    csv_files = sorted(target.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV found in the downloaded dataset at {target}")

    corpus = LoadedCorpus(tenant_id=tenant_id, source=f"kaggle:{dataset}")
    # CSV fields in text datasets routinely exceed the default 128KB limit.
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or text_column not in reader.fieldnames:
                logger.warning(
                    "Column '%s' not in %s (available: %s)",
                    text_column, csv_path.name, reader.fieldnames,
                )
                continue

            for index, row in enumerate(reader):
                if len(corpus.documents) >= max_rows:
                    break
                text = (row.get(text_column) or "").strip()
                if len(text) < min_chars:
                    continue
                raw_title = (row.get(title_column) or "").strip() if title_column else ""
                slug = re.sub(r"[^a-zA-Z0-9]+", "_", raw_title).strip("_").lower()[:60]
                doc_id = f"kaggle_{slug or index}"
                if doc_id in corpus.documents:
                    doc_id = f"{doc_id}_{index}"
                body = f"# {raw_title}\n\n{text}" if raw_title else text
                corpus.documents[doc_id] = body
                corpus.metadata[doc_id] = {
                    "source": f"kaggle:{dataset}",
                    "title": raw_title or doc_id,
                    "file": csv_path.name,
                    "chars": len(body),
                }
        if len(corpus.documents) >= max_rows:
            break

    return corpus


# ------------------------------------------------------------------------- Local

def load_local(tenant_id: str, directory: Path, max_files: int = 200) -> LoadedCorpus:
    """Build a corpus from a directory of .txt / .md files."""
    root = Path(directory)
    if not root.is_dir():
        raise RuntimeError(f"Corpus directory not found: {root}")

    corpus = LoadedCorpus(tenant_id=tenant_id, source=f"local:{root}")
    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in (".txt", ".md")])

    for path in files[:max_files]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s: %s", path, exc)
            continue
        if len(text) < 200:
            continue
        doc_id = f"local_{re.sub(r'[^a-zA-Z0-9]+', '_', path.stem).strip('_').lower()}"
        corpus.documents[doc_id] = text
        corpus.metadata[doc_id] = {
            "source": "local", "title": path.stem, "path": str(path), "chars": len(text)
        }

    return corpus


# ------------------------------------------------------------------------ Caching

def cache_corpus(corpus: LoadedCorpus, cache_dir: Path) -> Path:
    """Persist a corpus so later runs do not re-fetch it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{corpus.tenant_id}_corpus.json"
    path.write_text(
        json.dumps(
            {
                "tenant_id": corpus.tenant_id,
                "source": corpus.source,
                "documents": corpus.documents,
                "metadata": corpus.metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load_cached_corpus(tenant_id: str, cache_dir: Path) -> Optional[LoadedCorpus]:
    """Load a previously cached corpus, or None if absent."""
    path = Path(cache_dir) / f"{tenant_id}_corpus.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return LoadedCorpus(
        tenant_id=data["tenant_id"],
        source=data.get("source", "cache"),
        documents=data.get("documents", {}),
        metadata=data.get("metadata", {}),
    )


def load_corpus(
    tenant_id: str,
    source: str = "wikipedia",
    cache_dir: Path = Path("data/corpus_cache"),
    use_cache: bool = True,
    **kwargs: Any,
) -> LoadedCorpus:
    """Load a corpus from the named source, using the cache when available."""
    if use_cache:
        cached = load_cached_corpus(tenant_id, cache_dir)
        if cached and cached.documents:
            logger.info(
                "Using cached corpus for '%s' (%d documents).", tenant_id, cached.document_count
            )
            return cached

    if source == "wikipedia":
        corpus = load_wikipedia(tenant_id, **kwargs)
    elif source == "kaggle":
        corpus = load_kaggle(tenant_id, **kwargs)
    elif source == "local":
        corpus = load_local(tenant_id, **kwargs)
    else:
        raise ValueError(f"Unknown corpus source: {source!r}")

    if corpus.documents:
        cache_corpus(corpus, cache_dir)
    return corpus

"""Universal document loader: routes any source format to the right pipeline.

Two pipelines exist, and picking the wrong one wastes work or loses information:

  structured   CSV, TSV, XLSX, JSON, JSONL. Entities are already separated into
               columns, so relations are *read* rather than inferred and carry
               confidence 1.0. Running NER over a cell would be strictly worse
               than reading its column header.

  prose        TXT, MD, PDF, DOCX, HTML. Needs the full chunk -> NER -> relation
               extraction path, where relations are probabilistic.

Semi-structured documents (a PDF containing tables, a Markdown file with a table)
are split: tables go through the structured path, surrounding prose through the
prose path, and they merge on shared entity ids.

PDF and DOCX extraction is delegated to Docling when installed, which handles
layout and table structure properly. Without it those formats are unavailable
rather than silently mis-parsed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

STRUCTURED_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl"}
PROSE_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}
RICH_SUFFIXES = {".pdf", ".docx", ".pptx", ".html", ".htm"}


@dataclass
class LoadedDocument:
    """One document ready for ingestion, tagged with the pipeline it needs."""

    doc_id: str
    text: str
    pipeline: str                       # "structured" | "prose"
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    """Detects source format and produces documents for the correct pipeline."""

    @staticmethod
    def classify(path: Path) -> str:
        """Return the pipeline a file needs: structured, prose, or rich."""
        suffix = path.suffix.lower()
        if suffix in STRUCTURED_SUFFIXES:
            return "structured"
        if suffix in PROSE_SUFFIXES:
            return "prose"
        if suffix in RICH_SUFFIXES:
            return "rich"
        return "unsupported"

    # ------------------------------------------------------------------ prose
    def load_prose(self, path: Path) -> List[LoadedDocument]:
        """Read a plain-text or markdown file as a single document."""
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return []
        doc_id = f"doc_{path.stem.lower().replace(' ', '_')}"
        return [
            LoadedDocument(
                doc_id=doc_id,
                text=text,
                pipeline="prose",
                metadata={"source": str(path), "format": path.suffix.lstrip(".")},
            )
        ]

    # ------------------------------------------------------------------ rich
    def load_rich(self, path: Path) -> List[LoadedDocument]:
        """Extract PDF/DOCX/HTML via Docling, preserving table structure.

        Docling converts to markdown with tables intact, so the chunker's heading
        awareness works and tables remain recognizable for the structured path.
        """
        try:
            from docling.document_converter import DocumentConverter  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                f"Reading {path.suffix} requires Docling. Install with: pip install docling\n"
                "Alternatively convert the file to .txt, .md, or .csv first."
            ) from exc

        converter = DocumentConverter()
        result = converter.convert(str(path))
        markdown = result.document.export_to_markdown()
        if not markdown.strip():
            return []

        doc_id = f"doc_{path.stem.lower().replace(' ', '_')}"
        return [
            LoadedDocument(
                doc_id=doc_id,
                text=markdown,
                pipeline="prose",
                metadata={
                    "source": str(path),
                    "format": path.suffix.lstrip("."),
                    "extractor": "docling",
                },
            )
        ]

    # ------------------------------------------------------------------ public
    def load(self, path: Path, max_rows: Optional[int] = None) -> Tuple[str, List[LoadedDocument]]:
        """Load any supported file. Returns (pipeline, documents).

        For structured sources the caller should use `structured_ingestion_service`
        directly instead, since it produces entities and relations alongside text —
        this path returns only the verbalized rows.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {path}")

        kind = self.classify(path)

        if kind == "structured":
            from app.services.structured_ingestion import (  # noqa: PLC0415
                structured_ingestion_service,
            )

            rows, _, profiles = structured_ingestion_service.analyze(path, max_rows)
            documents = [
                LoadedDocument(
                    doc_id=f"row_{index}",
                    text=structured_ingestion_service.verbalize_row(row, profiles),
                    pipeline="structured",
                    metadata={"source": str(path), "row_index": index},
                )
                for index, row in enumerate(rows)
            ]
            return "structured", [d for d in documents if d.text.strip()]

        if kind == "prose":
            return "prose", self.load_prose(path)

        if kind == "rich":
            return "prose", self.load_rich(path)

        raise ValueError(
            f"Unsupported format '{path.suffix}'. Supported: "
            f"{sorted(STRUCTURED_SUFFIXES | PROSE_SUFFIXES | RICH_SUFFIXES)}"
        )

    def load_directory(
        self, directory: Path, max_files: int = 500
    ) -> Dict[str, List[LoadedDocument]]:
        """Load every supported file in a directory, grouped by pipeline."""
        directory = Path(directory)
        grouped: Dict[str, List[LoadedDocument]] = {"structured": [], "prose": []}

        for path in sorted(directory.rglob("*"))[:max_files]:
            if not path.is_file() or self.classify(path) == "unsupported":
                continue
            try:
                pipeline, documents = self.load(path)
                grouped[pipeline].extend(documents)
            except Exception as exc:  # noqa: BLE001 - one bad file must not abort
                logger.warning("Skipping %s: %s", path.name, exc)

        return grouped


document_loader = DocumentLoader()

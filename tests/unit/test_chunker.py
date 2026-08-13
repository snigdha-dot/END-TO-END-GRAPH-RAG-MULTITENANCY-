"""Unit tests for structure-aware chunking with overlap and hierarchy."""
from __future__ import annotations

from app.services.chunking_service import ChunkingService, chunking_service

MOVIE_DOC = """
# Inception

Inception is a 2010 science fiction action film written and directed by
Christopher Nolan. It stars Leonardo DiCaprio as a professional thief.

## Plot

Dom Cobb extracts secrets from within the subconscious during the dream state.
He is offered a chance to have his criminal history erased.

## Production

Nolan wrote the screenplay over nine years. Warner Bros produced the film.
Hans Zimmer composed the score.

## Reception

The film grossed over 836 million dollars worldwide and won four Academy Awards.
"""


def test_document_splits_into_chunks():
    chunks = chunking_service.chunk_document("inception_wiki", MOVIE_DOC)
    assert len(chunks) >= 1
    assert all(c.parent_doc_id == "inception_wiki" for c in chunks)
    assert all(c.token_count > 0 for c in chunks)


def test_section_hierarchy_is_captured():
    """Markdown headings become a breadcrumb so chunks are interpretable alone."""
    chunks = chunking_service.chunk_document("inception_wiki", MOVIE_DOC)
    paths = [tuple(c.section_path) for c in chunks]
    assert any("Inception" in p for p in paths if p)


def test_contextual_text_prefixes_the_breadcrumb():
    chunks = chunking_service.chunk_document("inception_wiki", MOVIE_DOC)
    sectioned = [c for c in chunks if c.section_path]
    if sectioned:
        contextual = sectioned[0].contextual_text()
        assert sectioned[0].section_path[0] in contextual


def test_sibling_links_form_a_chain():
    """Parent-child linkage: each chunk knows its neighbours."""
    chunks = chunking_service.chunk_document("doc", MOVIE_DOC)
    if len(chunks) > 1:
        assert chunks[0].prev_chunk_id is None
        assert chunks[0].next_chunk_id == chunks[1].chunk_id
        assert chunks[-1].next_chunk_id is None
        assert chunks[-1].prev_chunk_id == chunks[-2].chunk_id


def test_overlap_carries_context_across_boundaries():
    """Plan section 3: 100-token overlap, so a fact spanning a boundary survives."""
    service = ChunkingService(target_tokens=40, max_tokens=60, overlap_tokens=15)
    content = "\n\n".join(f"Paragraph {i} contains distinctive content about topic {i}." * 3
                          for i in range(8))
    chunks = service.chunk_document("overlap_doc", content)
    assert len(chunks) > 1
    assert any(c.has_overlap for c in chunks[1:])


def test_zero_overlap_is_respected():
    service = ChunkingService(target_tokens=40, max_tokens=60, overlap_tokens=0)
    content = "\n\n".join(f"Paragraph {i} with content." * 5 for i in range(6))
    chunks = service.chunk_document("no_overlap", content)
    assert all(not c.has_overlap for c in chunks)


def test_empty_document_yields_no_chunks():
    assert chunking_service.chunk_document("empty", "") == []
    assert chunking_service.chunk_document("blank", "   \n\n  ") == []


def test_oversized_paragraph_is_split_on_sentences():
    service = ChunkingService(target_tokens=30, max_tokens=40, overlap_tokens=0)
    giant = " ".join(f"This is sentence number {i} in a very long paragraph." for i in range(40))
    chunks = service.chunk_document("giant", giant)
    assert len(chunks) > 1


def test_chunk_ids_are_unique_and_ordered():
    chunks = chunking_service.chunk_document("doc", MOVIE_DOC)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

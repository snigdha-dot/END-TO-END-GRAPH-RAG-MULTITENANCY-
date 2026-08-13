"""Structure-aware semantic chunking service."""
import re
from typing import Any, Dict, List
from pydantic import BaseModel

class DocumentChunk(BaseModel):
    chunk_id: str
    parent_doc_id: str
    chunk_index: int
    text: str
    token_count: int
    metadata: Dict[str, Any]


class ChunkingService:
    def __init__(self, target_chunk_size: int = 500, overlap: int = 100):
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap

    def chunk_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Split raw text into structure-aware semantic chunks."""
        meta = metadata or {}
        
        # 1. Split into paragraphs / markdown sections
        raw_sections = re.split(r'\n\s*\n', content)
        
        chunks: List[DocumentChunk] = []
        current_chunk_text = ""
        chunk_idx = 0

        for section in raw_sections:
            section_text = section.strip()
            if not section_text:
                continue

            # Estimate token count (~4 chars per token)
            sec_tokens = len(section_text) // 4

            if len(current_chunk_text) // 4 + sec_tokens <= self.target_chunk_size:
                current_chunk_text += ("\n\n" if current_chunk_text else "") + section_text
            else:
                if current_chunk_text:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                            parent_doc_id=doc_id,
                            chunk_index=chunk_idx,
                            text=current_chunk_text,
                            token_count=len(current_chunk_text) // 4,
                            metadata=meta
                        )
                    )
                    chunk_idx += 1
                current_chunk_text = section_text

        if current_chunk_text:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                    parent_doc_id=doc_id,
                    chunk_index=chunk_idx,
                    text=current_chunk_text,
                    token_count=len(current_chunk_text) // 4,
                    metadata=meta
                )
            )

        return chunks


chunking_service = ChunkingService()

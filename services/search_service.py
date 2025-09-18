

from __future__ import annotations
from typing import Optional

from services.models import QARecord, QueryRequest
from repo import repo_qa_find_by_hash


class SearchService:
    """Search service with exact, fuzzy, and semantic modes (fuzzy/semantic are placeholders)."""

    def search_exact(self, request: QueryRequest) -> Optional[QARecord]:
        """Perform exact search using file_hash + question_hash."""
        if not request.file_hash:
            raise ValueError("QueryRequest.file_hash is required for exact search")
        return repo_qa_find_by_hash(file_hash=request.file_hash, question_hash=request.question_hash)

    def search_fuzzy(self, request: QueryRequest):
        """Placeholder for fuzzy search (e.g., edit distance or trigram similarity)."""
        raise NotImplementedError("Fuzzy search not yet implemented")

    def search_semantic(self, request: QueryRequest):
        """Placeholder for semantic search using embeddings."""
        raise NotImplementedError("Semantic search not yet implemented")
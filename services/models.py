# services/models.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

@dataclass(slots=True)
class MatchResults:
    mode: str  # "exact" | "fuzzy" | "semantic"
    records: List[QARecord]
    scores: Optional[List[float]] = None
    top_k: int = 5
    took_ms: float = 0.0

# Persistentes DB-Modell (qa_pairs)
@dataclass(slots=True)
class QARecord:
    id: Optional[int] = None
    question: str = ""
    question_norm: str = ""
    question_hash: str = ""
    answer: Optional[str] = None
    file_hash: Optional[str] = None
    embedding: Optional[List[float]] = None
    embed_model: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

# Transientes Request-Objekt für eine Anfrage
@dataclass(slots=True)
class QueryRequest:
    question_raw: str
    question_norm: str
    question_hash: str
    file_hash: Optional[str] = None
    top_k: int = 5
    created_at: datetime = field(default_factory=datetime.utcnow)
    token_usage: Optional[TokenUsage] = None
    decision: Optional[str] = None  # "exact" | "fuzzy" | "semantic" | "llm"
    badges: List[str] = field(default_factory=list)
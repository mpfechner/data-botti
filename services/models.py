# services/models.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

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
import re
import unicodedata
import hashlib
from typing import Optional, Dict, Any

from repo import (
    repo_qa_insert,
    repo_qa_find_by_hash,
    repo_qa_save_embedding,
    repo_embeddings_by_file,
)

from sentence_transformers import SentenceTransformer
import numpy as np
import threading

from services.models import QueryRequest, QARecord

MODEL_NAME = "distiluse-base-multilingual-cased-v2"

_model: Optional[SentenceTransformer] = None
_model_lock = threading.Lock()

def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_question(text_norm: str) -> np.ndarray:
    model = get_embedding_model()
    return model.encode([text_norm])[0]


def save_embedding(qa_id: int, vec: np.ndarray, model: str = MODEL_NAME) -> None:
    """Persist an embedding vector for a QA row. Ensures float32 dtype and stores raw bytes."""
    if vec is None:
        raise ValueError("vec is None")
    v = np.asarray(vec, dtype=np.float32)
    repo_qa_save_embedding(qa_id=qa_id, model=model, dim=int(v.shape[0]), vec=v.tobytes())


def find_semantic_candidates(file_hash: str, query_vec: np.ndarray, k: int = 20, model: str = MODEL_NAME):
    """Naive in-memory cosine similarity over stored embeddings for the given file_hash.
    Returns list[(qa_id, score)] sorted by score desc. Placeholder until k-NN index exists."""
    q = np.asarray(query_vec, dtype=np.float32)
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        return []
    rows = repo_embeddings_by_file(file_hash=file_hash, model=model)
    candidates = []
    for r in rows:
        qa_id = r[0]
        dim = int(r[1])
        vec_bytes = r[2]
        v = np.frombuffer(vec_bytes, dtype=np.float32, count=dim)
        denom = (q_norm * float(np.linalg.norm(v)))
        score = 0.0 if denom == 0.0 else float(np.dot(q, v) / denom)
        candidates.append((qa_id, score))
    candidates.sort(key=lambda t: t[1], reverse=True)
    return candidates[:k]



def normalize_question(text: str) -> str:
    """
    Normalize a question string:
    - Unicode NFKC normalization
    - lowercasing
    - collapse whitespace
    - strip leading/trailing spaces
    - unify basic punctuation spacing
    """
    if not text:
        return ""
    norm = unicodedata.normalize("NFKC", text)
    norm = norm.lower().strip()
    norm = re.sub(r"\s+", " ", norm)
    norm = re.sub(r"\s*([?.!,;:])\s*", r"\1 ", norm)
    return norm.strip()


def hash_question(text_norm: str) -> str:
    """
    Compute SHA-256 hash of normalized question.
    Returns hex digest string.
    """
    if not text_norm:
        return ""
    return hashlib.sha256(text_norm.encode("utf-8")).hexdigest()


def save_qa(
    file_hash: str,
    question_original: str,
    question_norm: str,
    question_hash: str,
    answer: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Save a Q&A pair into the database via repo.
    Returns new qa_pairs.id
    """
    return repo_qa_insert(
        file_hash=file_hash,
        question_original=question_original,
        question_norm=question_norm,
        question_hash=question_hash,
        answer=answer,
        meta=meta,
    )


def find_exact_qa(file_hash: str, question_hash: str) -> Optional[QARecord]:
    """
    Look up exact match in qa_pairs by file_hash + question_hash.
    Returns QARecord or None.
    """
    return repo_qa_find_by_hash(file_hash=file_hash, question_hash=question_hash)


def make_query_request(question_raw: str, file_hash: str | None = None) -> QueryRequest:
    """Build a QueryRequest dataclass from raw question + file_hash."""
    q_norm = normalize_question(question_raw)
    q_hash = hash_question(q_norm)
    return QueryRequest(
        question_raw=question_raw,
        question_norm=q_norm,
        question_hash=q_hash,
        file_hash=file_hash,
    )

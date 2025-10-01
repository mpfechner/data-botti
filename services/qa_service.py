import re
import unicodedata
import hashlib
import logging
from typing import Optional, Dict, Any

from repo import (
    repo_qa_insert,
    repo_qa_find_by_hash,
    repo_qa_save_embedding,
    repo_embeddings_by_file,
    repo_qa_find_by_id,
)

import numpy as np

from services.embeddings import embed_query, embed_passage

from services.ai_client import ask_model
from services.models import QueryRequest, QARecord, TokenUsage, MatchResults

MODEL_NAME = "intfloat/multilingual-e5-base"
logger = logging.getLogger(__name__)

def embedding_exists_for_qa(file_hash: str, qa_id: int, *, model: str = MODEL_NAME) -> bool:
    rows = repo_embeddings_by_file(file_hash=file_hash, model=model)
    for r in rows:
        existing_qa_id = r[0]
        if int(existing_qa_id) == int(qa_id):
            return True
    return False

def upsert_embedding_for_qa(file_hash: str, qa_id: int, question_norm: str, *, model: str = MODEL_NAME) -> bool:
    """Create (or idempotently upsert) an embedding for the given QA.
    Embedding failures are logged as WARN and do not raise.
    Returns True on success, False if embedding/persist failed.
    """
    try:
        if embedding_exists_for_qa(file_hash, qa_id, model=model):
            return True
        vec = embed_question(question_norm)
        save_embedding(qa_id=qa_id, vec=vec, model=model)
        return True
    except Exception as e:
        # Do not block the request flow on embedding errors
        logger.warning(
            "embedding_failed",
            extra={
                "qa_id": qa_id,
                "model": model,
                "error_type": type(e).__name__,
                "error_msg": str(e),
            },
        )
        return False

def embed_question(text_norm: str) -> np.ndarray:
    vec = embed_query(text_norm)
    return np.asarray(vec, dtype=np.float32)


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
    Returns new qa_pairs.id and synchronously attempts to persist an embedding
    for the normalized question. Embedding failures are logged but do not
    affect the QA insert.
    """
    qa_id = repo_qa_insert(
        file_hash=file_hash,
        question_original=question_original,
        question_norm=question_norm,
        question_hash=question_hash,
        answer=answer,
        meta=meta,
    )
    # Always attempt to embed after a successful save; do not block on errors
    upsert_embedding_for_qa(file_hash=file_hash, qa_id=qa_id, question_norm=question_norm)
    return qa_id

def backfill_missing_embeddings(file_hash: str | None = None, *, model: str = MODEL_NAME, batch_size: int = 200) -> None:
    """Iterate through qa_pairs without embedding and create embeddings synchronously.
    Logs progress. Errors are logged but do not interrupt the batch run.
    """
    from repo import repo_qa_without_embedding

    rows = repo_qa_without_embedding(file_hash=file_hash, model=model)
    total = len(rows)
    logger.info("backfill_start", extra={"file_hash": file_hash, "total": total})

    processed = 0
    for r in rows:
        qa_id = r["id"]
        fh = r["file_hash"]
        qn = r["question_norm"]
        ok = upsert_embedding_for_qa(file_hash=fh, qa_id=qa_id, question_norm=qn, model=model)
        processed += 1
        if processed % batch_size == 0:
            logger.info("backfill_progress", extra={"processed": processed, "total": total})

    logger.info("backfill_done", extra={"file_hash": file_hash, "processed": processed, "total": total})


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

def call_llm_and_record(request: QueryRequest, *, model: str = "gpt-4o-mini", max_tokens: int = 800, temperature: float | None = None) -> str:
    """Call the LLM via ai_client.ask_model and attach TokenUsage to the QueryRequest."""
    content, usage = ask_model(
        prompt=request.question_raw,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    request.token_usage = usage
    return content


# New orchestrate function
def orchestrate(request: QueryRequest) -> MatchResults:
    """Orchestrate QA flow: try exact, else call LLM and save new QA."""
    # Try exact match first
    if request.file_hash:
        rec = find_exact_qa(file_hash=request.file_hash, question_hash=request.question_hash)
        if rec:
            request.decision = "exact"
            request.badges.append("exact")
            return MatchResults(mode="exact", records=[rec], top_k=1, took_ms=0.0)

    # Try semantic search
    if request.file_hash:
        q_vec = embed_query(request.question_norm)
        candidates = find_semantic_candidates(request.file_hash, q_vec)
        if candidates:
            best_id, best_score = candidates[0]
            if best_score > 0.75:  # simple threshold
                rec = repo_qa_find_by_id(qa_id=best_id)
                if rec:
                    request.decision = "semantic"
                    request.badges.append("semantic")
                    return MatchResults(mode="semantic", records=[rec], top_k=1, took_ms=0.0)

    # Fallback: call LLM and save new QA
    answer = call_llm_and_record(request)
    qa_id = save_qa(
        file_hash=request.file_hash or "unknown",
        question_original=request.question_raw,
        question_norm=request.question_norm,
        question_hash=request.question_hash,
        answer=answer,
        meta={"source": "orchestrate"},
    )
    rec_new = QARecord(
        id=qa_id,
        question=request.question_raw,
        question_norm=request.question_norm,
        question_hash=request.question_hash,
        answer=answer,
        file_hash=request.file_hash,
    )
    request.decision = "llm"
    request.badges.append("llm")
    return MatchResults(mode="llm", records=[rec_new], top_k=1, took_ms=0.0)

import re
import unicodedata
import hashlib
from typing import Optional, Dict, Any

from repo import repo_qa_insert, repo_qa_find_by_hash


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


def find_exact_qa(file_hash: str, question_hash: str) -> Optional[Dict[str, Any]]:
    """
    Look up exact match in qa_pairs by file_hash + question_hash.
    Returns dict row or None.
    """
    return repo_qa_find_by_hash(file_hash=file_hash, question_hash=question_hash)

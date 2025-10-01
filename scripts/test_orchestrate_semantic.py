from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pytest
import os, hashlib

def _new_test_file_hash() -> str:
    seed = b"pytest-" + os.urandom(16)
    return hashlib.sha256(seed).hexdigest()

from services.qa_service import make_query_request, orchestrate, save_qa
from app import app as flask_app
from repo import repo_embeddings_delete_by_qa, repo_qa_delete_by_id


def test_orchestrate_semantic_flow():
    with flask_app.app_context():
        file_hash = _new_test_file_hash()
        # ensure a base QA exists
        q_text = "What is the capital of France?"
        import hashlib as _hl
        q_norm = q_text.lower()
        q_hash = _hl.sha256(q_norm.encode("utf-8")).hexdigest()
        base_id = save_qa(
            file_hash=file_hash,
            question_original=q_text,
            question_norm=q_norm,
            question_hash=q_hash,
            answer="Paris",
            meta={"test": True},
        )

        # now ask a semantically similar but not exact question
        req = make_query_request("Please tell me the capital city of France", file_hash=file_hash)
        res = orchestrate(req)

        assert res.mode == "semantic"
        assert req.decision == "semantic"
        assert res.records[0].answer == "Paris"

        # --- cleanup: remove created embeddings and QA row ---
        MODEL = "intfloat/multilingual-e5-base"
        repo_embeddings_delete_by_qa(qa_id=base_id, model=MODEL)
        repo_qa_delete_by_id(qa_id=base_id)